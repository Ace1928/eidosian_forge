import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class CenterMass(AFNICommandBase):
    """Computes center of mass using 3dCM command

    .. note::

      By default, the output is (x,y,z) values in DICOM coordinates. But
      as of Dec, 2016, there are now command line switches for other options.


    For complete details, see the `3dCM Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dCM.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> cm = afni.CenterMass()
    >>> cm.inputs.in_file = 'structural.nii'
    >>> cm.inputs.cm_file = 'cm.txt'
    >>> cm.inputs.roi_vals = [2, 10]
    >>> cm.cmdline
    '3dCM -roi_vals 2 10 structural.nii > cm.txt'
    >>> res = 3dcm.run()  # doctest: +SKIP

    """
    _cmd = '3dCM'
    input_spec = CenterMassInputSpec
    output_spec = CenterMassOutputSpec

    def _list_outputs(self):
        outputs = super(CenterMass, self)._list_outputs()
        outputs['out_file'] = os.path.abspath(self.inputs.in_file)
        outputs['cm_file'] = os.path.abspath(self.inputs.cm_file)
        sout = np.loadtxt(outputs['cm_file'], ndmin=2)
        outputs['cm'] = [tuple(s) for s in sout]
        return outputs