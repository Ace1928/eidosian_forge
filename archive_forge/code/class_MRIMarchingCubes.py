import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIMarchingCubes(FSCommand):
    """
    Uses Freesurfer's mri_mc to create surfaces by tessellating a given input volume

    Example
    -------

    >>> import nipype.interfaces.freesurfer as fs
    >>> mc = fs.MRIMarchingCubes()
    >>> mc.inputs.in_file = 'aseg.mgz'
    >>> mc.inputs.label_value = 17
    >>> mc.inputs.out_file = 'lh.hippocampus'
    >>> mc.run() # doctest: +SKIP
    """
    _cmd = 'mri_mc'
    input_spec = MRIMarchingCubesInputSpec
    output_spec = MRIMarchingCubesOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['surface'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            return os.path.abspath(self.inputs.out_file)
        else:
            _, name, ext = split_filename(self.inputs.in_file)
            return os.path.abspath(name + ext + '_' + str(self.inputs.label_value))