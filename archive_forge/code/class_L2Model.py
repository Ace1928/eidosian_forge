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
class L2Model(BaseInterface):
    """Generate subject specific second level model

    Examples
    --------

    >>> from nipype.interfaces.fsl import L2Model
    >>> model = L2Model(num_copes=3) # 3 sessions

    """
    input_spec = L2ModelInputSpec
    output_spec = L2ModelOutputSpec

    def _run_interface(self, runtime):
        cwd = os.getcwd()
        mat_txt = ['/NumWaves   1', '/NumPoints  {:d}'.format(self.inputs.num_copes), '/PPheights  1', '', '/Matrix']
        for i in range(self.inputs.num_copes):
            mat_txt += ['1']
        mat_txt = '\n'.join(mat_txt)
        con_txt = ['/ContrastName1  group mean', '/NumWaves   1', '/NumContrasts   1', '/PPheights  1', '/RequiredEffect     100', '', '/Matrix', '1']
        con_txt = '\n'.join(con_txt)
        grp_txt = ['/NumWaves   1', '/NumPoints  {:d}'.format(self.inputs.num_copes), '', '/Matrix']
        for i in range(self.inputs.num_copes):
            grp_txt += ['1']
        grp_txt = '\n'.join(grp_txt)
        txt = {'design.mat': mat_txt, 'design.con': con_txt, 'design.grp': grp_txt}
        for i, name in enumerate(['design.mat', 'design.con', 'design.grp']):
            f = open(os.path.join(cwd, name), 'wt')
            f.write(txt[name])
            f.close()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        for field in list(outputs.keys()):
            outputs[field] = os.path.join(os.getcwd(), field.replace('_', '.'))
        return outputs