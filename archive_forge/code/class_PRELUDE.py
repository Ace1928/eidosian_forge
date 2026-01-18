import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PRELUDE(FSLCommand):
    """FSL prelude wrapper for phase unwrapping

    Examples
    --------

    Please insert examples for use of this command

    """
    input_spec = PRELUDEInputSpec
    output_spec = PRELUDEOutputSpec
    _cmd = 'prelude'

    def __init__(self, **kwargs):
        super(PRELUDE, self).__init__(**kwargs)
        warn('This has not been fully tested. Please report any failures.')

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.unwrapped_phase_file
        if not isdefined(out_file):
            if isdefined(self.inputs.phase_file):
                out_file = self._gen_fname(self.inputs.phase_file, suffix='_unwrapped')
            elif isdefined(self.inputs.complex_phase_file):
                out_file = self._gen_fname(self.inputs.complex_phase_file, suffix='_phase_unwrapped')
        outputs['unwrapped_phase_file'] = os.path.abspath(out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'unwrapped_phase_file':
            return self._list_outputs()['unwrapped_phase_file']
        return None