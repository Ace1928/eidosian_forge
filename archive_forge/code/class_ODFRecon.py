import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
class ODFRecon(CommandLine):
    """Use odf_recon to generate tensors and other maps"""
    input_spec = ODFReconInputSpec
    output_spec = ODFReconOutputSpec
    _cmd = 'odf_recon'

    def _list_outputs(self):
        out_prefix = self.inputs.out_prefix
        output_type = self.inputs.output_type
        outputs = self.output_spec().get()
        outputs['B0'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_b0.' + output_type))
        outputs['DWI'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_dwi.' + output_type))
        outputs['max'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_max.' + output_type))
        outputs['ODF'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_odf.' + output_type))
        if isdefined(self.inputs.output_entropy):
            outputs['entropy'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_entropy.' + output_type))
        return outputs