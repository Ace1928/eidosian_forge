import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
from adagio.instances import TaskContext
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_function, get_full_type_path
def _is_union(anno: Any):
    return _get_origin_type(anno, False) is Union