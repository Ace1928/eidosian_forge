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
class FILMGLS(FSLCommand):
    """Use FSL film_gls command to fit a design matrix to voxel timeseries

    Examples
    --------

    Initialize with no options, assigning them when calling run:

    >>> from nipype.interfaces import fsl
    >>> fgls = fsl.FILMGLS()
    >>> res = fgls.run('in_file', 'design_file', 'thresh', rn='stats') #doctest: +SKIP

    Assign options through the ``inputs`` attribute:

    >>> fgls = fsl.FILMGLS()
    >>> fgls.inputs.in_file = 'functional.nii'
    >>> fgls.inputs.design_file = 'design.mat'
    >>> fgls.inputs.threshold = 10
    >>> fgls.inputs.results_dir = 'stats'
    >>> res = fgls.run() #doctest: +SKIP

    Specify options when creating an instance:

    >>> fgls = fsl.FILMGLS(in_file='functional.nii', design_file='design.mat', threshold=10, results_dir='stats')
    >>> res = fgls.run() #doctest: +SKIP

    """
    _cmd = 'film_gls'
    input_spec = FILMGLSInputSpec
    output_spec = FILMGLSOutputSpec
    if Info.version() and LooseVersion(Info.version()) > LooseVersion('5.0.6'):
        input_spec = FILMGLSInputSpec507
        output_spec = FILMGLSOutputSpec507
    elif Info.version() and LooseVersion(Info.version()) > LooseVersion('5.0.4'):
        input_spec = FILMGLSInputSpec505

    def _get_pe_files(self, cwd):
        files = None
        if isdefined(self.inputs.design_file):
            fp = open(self.inputs.design_file, 'rt')
            for line in fp.readlines():
                if line.startswith('/NumWaves'):
                    numpes = int(line.split()[-1])
                    files = []
                    for i in range(numpes):
                        files.append(self._gen_fname('pe%d.nii' % (i + 1), cwd=cwd))
                    break
            fp.close()
        return files

    def _get_numcons(self):
        numtcons = 0
        numfcons = 0
        if isdefined(self.inputs.tcon_file):
            fp = open(self.inputs.tcon_file, 'rt')
            for line in fp.readlines():
                if line.startswith('/NumContrasts'):
                    numtcons = int(line.split()[-1])
                    break
            fp.close()
        if isdefined(self.inputs.fcon_file):
            fp = open(self.inputs.fcon_file, 'rt')
            for line in fp.readlines():
                if line.startswith('/NumContrasts'):
                    numfcons = int(line.split()[-1])
                    break
            fp.close()
        return (numtcons, numfcons)

    def _list_outputs(self):
        outputs = self._outputs().get()
        cwd = os.getcwd()
        results_dir = os.path.join(cwd, self.inputs.results_dir)
        outputs['results_dir'] = results_dir
        pe_files = self._get_pe_files(results_dir)
        if pe_files:
            outputs['param_estimates'] = pe_files
        outputs['residual4d'] = self._gen_fname('res4d.nii', cwd=results_dir)
        outputs['dof_file'] = os.path.join(results_dir, 'dof')
        outputs['sigmasquareds'] = self._gen_fname('sigmasquareds.nii', cwd=results_dir)
        outputs['thresholdac'] = self._gen_fname('threshac1.nii', cwd=results_dir)
        if Info.version() and LooseVersion(Info.version()) < LooseVersion('5.0.7'):
            outputs['corrections'] = self._gen_fname('corrections.nii', cwd=results_dir)
        outputs['logfile'] = self._gen_fname('logfile', change_ext=False, cwd=results_dir)
        if Info.version() and LooseVersion(Info.version()) > LooseVersion('5.0.6'):
            pth = results_dir
            numtcons, numfcons = self._get_numcons()
            base_contrast = 1
            copes = []
            varcopes = []
            zstats = []
            tstats = []
            for i in range(numtcons):
                copes.append(self._gen_fname('cope%d.nii' % (base_contrast + i), cwd=pth))
                varcopes.append(self._gen_fname('varcope%d.nii' % (base_contrast + i), cwd=pth))
                zstats.append(self._gen_fname('zstat%d.nii' % (base_contrast + i), cwd=pth))
                tstats.append(self._gen_fname('tstat%d.nii' % (base_contrast + i), cwd=pth))
            if copes:
                outputs['copes'] = copes
                outputs['varcopes'] = varcopes
                outputs['zstats'] = zstats
                outputs['tstats'] = tstats
            fstats = []
            zfstats = []
            for i in range(numfcons):
                fstats.append(self._gen_fname('fstat%d.nii' % (base_contrast + i), cwd=pth))
                zfstats.append(self._gen_fname('zfstat%d.nii' % (base_contrast + i), cwd=pth))
            if fstats:
                outputs['fstats'] = fstats
                outputs['zfstats'] = zfstats
        return outputs