import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class ProductInterface(nib.BaseInterface):
    input_spec = ProductInputSpec
    output_spec = ProductOutputSpec

    def _run_interface(self, runtime):
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        global _products
        outputs = self._outputs().get()
        outputs['output1'] = self.inputs.input1 * self.inputs.input2
        _products.append(outputs['output1'])
        return outputs