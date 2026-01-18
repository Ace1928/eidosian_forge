import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
class ODFTracker(CommandLine):
    """Use odf_tracker to generate track file"""
    input_spec = ODFTrackerInputSpec
    output_spec = ODFTrackerOutputSpec
    _cmd = 'odf_tracker'

    def _run_interface(self, runtime):
        _, _, ext = split_filename(self.inputs.max)
        copyfile(self.inputs.max, os.path.abspath(self.inputs.input_data_prefix + '_max' + ext), copy=False)
        _, _, ext = split_filename(self.inputs.ODF)
        copyfile(self.inputs.ODF, os.path.abspath(self.inputs.input_data_prefix + '_odf' + ext), copy=False)
        return super(ODFTracker, self)._run_interface(runtime)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['track_file'] = os.path.abspath(self.inputs.out_file)
        return outputs