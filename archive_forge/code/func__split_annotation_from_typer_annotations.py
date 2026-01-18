import inspect
import sys
from copy import copy
from typing import Any, Callable, Dict, List, Tuple, Type, cast, get_type_hints
from typing_extensions import Annotated
from ._typing import get_args, get_origin
from .models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta
def _split_annotation_from_typer_annotations(base_annotation: Type[Any]) -> Tuple[Type[Any], List[ParameterInfo]]:
    if get_origin(base_annotation) is not Annotated:
        return (base_annotation, [])
    base_annotation, *maybe_typer_annotations = get_args(base_annotation)
    return (base_annotation, [annotation for annotation in maybe_typer_annotations if isinstance(annotation, ParameterInfo)])