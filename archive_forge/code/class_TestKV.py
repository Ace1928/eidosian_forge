from copy import deepcopy
from glob import glob
import os
import pytest
from ... import engine as pe
from .test_base import EngineTestInterface
import networkx
class TestKV(IOBase):
    _always_run = True
    output_spec = DynamicTraitedSpec

    def _list_outputs(self):
        outputs = {}
        outputs['test'] = 1
        outputs['foo'] = 'bar'
        return outputs