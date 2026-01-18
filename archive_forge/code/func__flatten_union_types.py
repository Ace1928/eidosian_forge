import datetime
import math
import typing as t
from wandb.util import (
def _flatten_union_types(wb_types: t.List[Type]) -> t.List[Type]:
    final_types = []
    for allowed_type in wb_types:
        if isinstance(allowed_type, UnionType):
            internal_types = _flatten_union_types(allowed_type.params['allowed_types'])
            for internal_type in internal_types:
                final_types.append(internal_type)
        else:
            final_types.append(allowed_type)
    return final_types