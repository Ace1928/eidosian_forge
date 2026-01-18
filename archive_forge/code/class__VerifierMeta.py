import inspect
import math
import operator
from collections.abc import Iterable
from typing import Any, Dict, final, List, Optional, Tuple, Type
import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt
class _VerifierMeta(type):
    _registry: Dict[str, Type['Verifier']] = {}

    def __new__(metacls, name, bases, attrs):
        if bases:
            if 'check' in attrs or '_check_graph_module' in attrs:
                raise SyntaxError('Overriding method check is not allowed.')
            assert 'dialect' in attrs and attrs['dialect'] != 'ATEN'
        else:
            assert 'check' in attrs
            assert '_check_graph_module' in attrs
            assert attrs['dialect'] == 'ATEN'
        assert isinstance(attrs['dialect'], str)
        ret = type.__new__(metacls, name, bases, attrs)
        metacls._registry[attrs['dialect']] = ret
        return ret