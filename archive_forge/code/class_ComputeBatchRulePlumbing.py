import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
@dataclass(frozen=True)
class ComputeBatchRulePlumbing:

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        opname = str(f.func.name)
        result = gen_vmap_plumbing(f)
        return result