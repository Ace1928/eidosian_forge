import os
from time import sleep
import nipype.interfaces.base as nib
import pytest
import nipype.pipeline.engine as pe
from nipype.pipeline.plugins.somaflow import soma_not_loaded
class SomaTestInterface(nib.BaseInterface):
    input_spec = InputSpec
    output_spec = OutputSpec

    def _run_interface(self, runtime):
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output1'] = [1, self.inputs.input1]
        return outputs