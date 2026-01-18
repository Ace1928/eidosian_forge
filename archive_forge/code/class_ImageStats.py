import os
from ..base import (
from ...utils.filemanip import split_filename
class ImageStats(CommandLine):
    """
    This program computes voxelwise statistics on a series of 3D images. The images
    must be in the same space; the operation is performed voxelwise and one output
    is produced per voxel.

    Examples
    --------

    >>> import nipype.interfaces.camino as cam
    >>> imstats = cam.ImageStats()
    >>> imstats.inputs.in_files = ['im1.nii','im2.nii','im3.nii']
    >>> imstats.inputs.stat = 'max'
    >>> imstats.run()                  # doctest: +SKIP
    """
    _cmd = 'imagestats'
    input_spec = ImageStatsInputSpec
    output_spec = ImageStatsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        output_root = self.inputs.output_root
        first_file = self.inputs.in_files[0]
        _, _, ext = split_filename(first_file)
        return output_root + ext