import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
def _get_output_warped_filename(self):
    if isdefined(self.inputs.print_out_composite_warp_file):
        return '--output [ %s, %d ]' % (self._gen_filename('output_image'), int(self.inputs.print_out_composite_warp_file))
    else:
        return '--output %s' % self._gen_filename('output_image')