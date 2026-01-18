import os
from ...utils.filemanip import split_filename
from ..base import (
def _gen_outputfile(self):
    outputfile = self.inputs.outputfile
    if not isdefined(outputfile):
        outputfile = self._gen_filename('outputfile')
    return outputfile