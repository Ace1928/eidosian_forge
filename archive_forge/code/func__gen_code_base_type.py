from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
from torchgen.api.types import Binding, CType, NamedCType
from torchgen.model import (
def _gen_code_base_type(self, arg_name: str, out_name: str, ctype: CType) -> Tuple[List[str], List[str]]:
    return ([f'{ctype.cpp_type()} {out_name} = {arg_name}.to<{ctype.cpp_type(strip_ref=True)}>();'], [])