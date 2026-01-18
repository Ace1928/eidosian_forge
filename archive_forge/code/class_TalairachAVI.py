import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class TalairachAVI(FSCommand):
    """
    Front-end for Avi Snyders image registration tool. Computes the
    talairach transform that maps the input volume to the MNI average_305.
    This does not add the xfm to the header of the input file. When called
    by recon-all, the xfm is added to the header after the transform is
    computed.

    Examples
    ========

    >>> from nipype.interfaces.freesurfer import TalairachAVI
    >>> example = TalairachAVI()
    >>> example.inputs.in_file = 'norm.mgz'
    >>> example.inputs.out_file = 'trans.mat'
    >>> example.cmdline
    'talairach_avi --i norm.mgz --xfm trans.mat'

    >>> example.run() # doctest: +SKIP
    """
    _cmd = 'talairach_avi'
    input_spec = TalairachAVIInputSpec
    output_spec = TalairachAVIOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        outputs['out_log'] = os.path.abspath('talairach_avi.log')
        outputs['out_txt'] = os.path.join(os.path.dirname(self.inputs.out_file), 'talsrcimg_to_' + str(self.inputs.atlas) + 't4_vox2vox.txt')
        return outputs