import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
from adagio.instances import TaskContext
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_function, get_full_type_path
def _parse_annotation(anno: Any) -> Dict[str, Any]:
    d = _try_parse(anno)
    if d is not None:
        return d
    assert_or_throw(anno.__module__ == 'typing', TypeError(f'{anno} is not a valid type'))
    nullable = False
    if _is_union(anno):
        tps = set(anno.__args__)
        if type(None) not in tps or len(tps) != 2:
            raise TypeError(f"{anno} can't be converted for TaskSpec")
        anno = [x for x in tps if x is not type(None)][0]
        nullable = True
    d = _try_parse(anno)
    if d is not None:
        d['nullable'] = nullable
        return d
    return dict(data_type=_get_origin_type(anno), nullable=nullable)