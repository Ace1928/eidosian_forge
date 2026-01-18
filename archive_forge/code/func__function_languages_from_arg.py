import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def _function_languages_from_arg(arg_proto: v2.program_pb2.Arg) -> Iterator[str]:
    which = arg_proto.WhichOneof('arg')
    if which == 'func':
        if arg_proto.func.type in ['add', 'mul']:
            yield 'linear'
            for a in arg_proto.func.args:
                yield from _function_languages_from_arg(a)
        if arg_proto.func.type in ['pow']:
            yield 'exp'
            for a in arg_proto.func.args:
                yield from _function_languages_from_arg(a)