import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MultipleRegressDesign(BaseInterface):
    """Generate multiple regression design

    .. note::
      FSL does not demean columns for higher level analysis.

    Please see `FSL documentation
    <http://www.fmrib.ox.ac.uk/fsl/feat5/detail.html#higher>`_
    for more details on model specification for higher level analysis.

    Examples
    --------

    >>> from nipype.interfaces.fsl import MultipleRegressDesign
    >>> model = MultipleRegressDesign()
    >>> model.inputs.contrasts = [['group mean', 'T',['reg1'],[1]]]
    >>> model.inputs.regressors = dict(reg1=[1, 1, 1], reg2=[2.,-4, 3])
    >>> model.run() # doctest: +SKIP

    """
    input_spec = MultipleRegressDesignInputSpec
    output_spec = MultipleRegressDesignOutputSpec

    def _run_interface(self, runtime):
        cwd = os.getcwd()
        regs = sorted(self.inputs.regressors.keys())
        nwaves = len(regs)
        npoints = len(self.inputs.regressors[regs[0]])
        ntcons = sum([1 for con in self.inputs.contrasts if con[1] == 'T'])
        nfcons = sum([1 for con in self.inputs.contrasts if con[1] == 'F'])
        mat_txt = ['/NumWaves       %d' % nwaves, '/NumPoints      %d' % npoints]
        ppheights = []
        for reg in regs:
            maxreg = np.max(self.inputs.regressors[reg])
            minreg = np.min(self.inputs.regressors[reg])
            if np.sign(maxreg) == np.sign(minreg):
                regheight = max([abs(minreg), abs(maxreg)])
            else:
                regheight = abs(maxreg - minreg)
            ppheights.append('%e' % regheight)
        mat_txt += ['/PPheights      ' + ' '.join(ppheights)]
        mat_txt += ['', '/Matrix']
        for cidx in range(npoints):
            mat_txt.append(' '.join(['%e' % self.inputs.regressors[key][cidx] for key in regs]))
        mat_txt = '\n'.join(mat_txt) + '\n'
        con_txt = []
        counter = 0
        tconmap = {}
        for conidx, con in enumerate(self.inputs.contrasts):
            if con[1] == 'T':
                tconmap[conidx] = counter
                counter += 1
                con_txt += ['/ContrastName%d   %s' % (counter, con[0])]
        con_txt += ['/NumWaves       %d' % nwaves, '/NumContrasts   %d' % ntcons, '/PPheights          %s' % ' '.join(['%e' % 1 for i in range(counter)]), '/RequiredEffect     %s' % ' '.join(['%.3f' % 100 for i in range(counter)]), '', '/Matrix']
        for idx in sorted(tconmap.keys()):
            convals = np.zeros((nwaves, 1))
            for regidx, reg in enumerate(self.inputs.contrasts[idx][2]):
                convals[regs.index(reg)] = self.inputs.contrasts[idx][3][regidx]
            con_txt.append(' '.join(['%e' % val for val in convals]))
        con_txt = '\n'.join(con_txt) + '\n'
        fcon_txt = ''
        if nfcons:
            fcon_txt = ['/NumWaves       %d' % ntcons, '/NumContrasts   %d' % nfcons, '', '/Matrix']
            for conidx, con in enumerate(self.inputs.contrasts):
                if con[1] == 'F':
                    convals = np.zeros((ntcons, 1))
                    for tcon in con[2]:
                        convals[tconmap[self.inputs.contrasts.index(tcon)]] = 1
                    fcon_txt.append(' '.join(['%d' % val for val in convals]))
            fcon_txt = '\n'.join(fcon_txt) + '\n'
        grp_txt = ['/NumWaves       1', '/NumPoints      %d' % npoints, '', '/Matrix']
        for i in range(npoints):
            if isdefined(self.inputs.groups):
                grp_txt += ['%d' % self.inputs.groups[i]]
            else:
                grp_txt += ['1']
        grp_txt = '\n'.join(grp_txt) + '\n'
        txt = {'design.mat': mat_txt, 'design.con': con_txt, 'design.fts': fcon_txt, 'design.grp': grp_txt}
        for key, val in list(txt.items()):
            if 'fts' in key and nfcons == 0:
                continue
            filename = key.replace('_', '.')
            f = open(os.path.join(cwd, filename), 'wt')
            f.write(val)
            f.close()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        nfcons = sum([1 for con in self.inputs.contrasts if con[1] == 'F'])
        for field in list(outputs.keys()):
            if 'fts' in field and nfcons == 0:
                continue
            outputs[field] = os.path.join(os.getcwd(), field.replace('_', '.'))
        return outputs