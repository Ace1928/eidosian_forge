import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
class StrPathConfuser(nib.SimpleInterface):
    input_spec = StrPathConfuserInputSpec
    output_spec = StrPathConfuserOutputSpec

    def _run_interface(self, runtime):
        out_path = os.path.abspath(os.path.basename(self.inputs.in_str) + '_path')
        open(out_path, 'w').close()
        self._results['out_str'] = self.inputs.in_str
        self._results['out_path'] = out_path
        self._results['out_tuple'] = (out_path, self.inputs.in_str)
        self._results['out_dict_path'] = {self.inputs.in_str: out_path}
        self._results['out_dict_str'] = {self.inputs.in_str: self.inputs.in_str}
        self._results['out_list'] = [self.inputs.in_str] * 2
        return runtime