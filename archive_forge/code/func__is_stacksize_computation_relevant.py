import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import (
import bytecode as _bytecode
from bytecode.concrete import ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import UNSET, Instr, Label, SetLineno, TryBegin, TryEnd
def _is_stacksize_computation_relevant(self, block_id: int, fingerprint: Tuple[int, Optional[bool]]) -> bool:
    if sys.version_info >= (3, 11):
        return fingerprint not in self.common.blocks_startsizes[block_id]
    elif (sizes := self.common.blocks_startsizes[block_id]):
        return fingerprint[0] > max((f[0] for f in sizes))
    else:
        return True