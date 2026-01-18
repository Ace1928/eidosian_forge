from typing import Callable, Iterable, Iterator, NoReturn, Union, TYPE_CHECKING
from typing_extensions import Protocol
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops.raw_types import Operation
def _bad_op_tree(root: OP_TREE) -> NoReturn:
    raise TypeError(f'Not an Operation or Iterable: {type(root)} {root}')