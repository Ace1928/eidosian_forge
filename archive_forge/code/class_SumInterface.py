import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class SumInterface(nib.SimpleInterface):
    input_spec = SumInputSpec
    output_spec = SumOutputSpec

    def _run_interface(self, runtime):
        global _sum
        global _sum_operands
        runtime.returncode = 0
        self._results['operands'] = self.inputs.input1
        self._results['output1'] = sum(self.inputs.input1)
        _sum_operands.append(self.inputs.input1)
        _sums.append(sum(self.inputs.input1))
        return runtime