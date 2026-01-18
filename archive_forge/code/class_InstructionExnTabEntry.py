import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
@dataclasses.dataclass
class InstructionExnTabEntry:
    start: 'Instruction'
    end: 'Instruction'
    target: 'Instruction'
    depth: int
    lasti: bool

    def __repr__(self) -> str:
        return f'InstructionExnTabEntry(start={self.start.short_inst_repr()}, end={self.end.short_inst_repr()}, target={self.target.short_inst_repr()}, depth={self.depth}, lasti={self.lasti})'

    def __eq__(self, o) -> bool:
        return self.start is o.start and self.end is o.end and (self.target is o.target) and (self.depth == o.depth) and (self.lasti == o.lasti)