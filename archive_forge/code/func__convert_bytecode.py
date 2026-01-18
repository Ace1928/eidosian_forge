import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
def _convert_bytecode(block, instlist, argnames: tuple[str, ...]) -> DDGBlock:
    converter = BC2DDG()
    if instlist[0].offset == 0:
        for arg in argnames:
            converter.load(f'var.{arg}')
    for inst in instlist:
        converter.convert(inst)
    return _converter_to_ddgblock(block, converter)