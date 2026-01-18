import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
def _gen_output_base(self):
    output_file_base = self.inputs.output_file_base
    if isdefined(output_file_base):
        return output_file_base
    else:
        base_file_name = os.path.split(self.inputs.input_file)[1]
        base_file_name_no_ext = os.path.splitext(base_file_name)[0]
        output_base = os.path.join(os.getcwd(), base_file_name_no_ext + '_bluroutput')
        return output_base