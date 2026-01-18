import datetime
import math
import typing as t
from wandb.util import (
def _union_assigner(allowed_types: t.List[Type], obj_or_type: t.Union[Type, t.Optional[t.Any]], type_mode=False) -> t.Union[t.List[Type], InvalidType]:
    resolved_types = []
    valid = False
    unknown_count = 0
    for allowed_type in allowed_types:
        if valid:
            resolved_types.append(allowed_type)
        elif isinstance(allowed_type, UnknownType):
            unknown_count += 1
        else:
            if type_mode:
                assert isinstance(obj_or_type, Type)
                assigned_type = allowed_type.assign_type(obj_or_type)
            else:
                assigned_type = allowed_type.assign(obj_or_type)
            if isinstance(assigned_type, InvalidType):
                resolved_types.append(allowed_type)
            else:
                resolved_types.append(assigned_type)
                valid = True
    if not valid:
        if unknown_count == 0:
            return InvalidType()
        else:
            if type_mode:
                assert isinstance(obj_or_type, Type)
                new_type = obj_or_type
            else:
                new_type = UnknownType().assign(obj_or_type)
            if isinstance(new_type, InvalidType):
                return InvalidType()
            else:
                resolved_types.append(new_type)
                unknown_count -= 1
    for _ in range(unknown_count):
        resolved_types.append(UnknownType())
    resolved_types = _flatten_union_types(resolved_types)
    resolved_types.sort(key=str)
    return resolved_types