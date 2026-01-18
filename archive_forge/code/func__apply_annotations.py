from __future__ import annotations as _annotations
import collections.abc
import dataclasses
import inspect
import re
import sys
import typing
import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from enum import Enum
from functools import partial
from inspect import Parameter, _ParameterKind, signature
from itertools import chain
from operator import attrgetter
from types import FunctionType, LambdaType, MethodType
from typing import (
from warnings import warn
from pydantic_core import CoreSchema, PydanticUndefined, core_schema, to_jsonable_python
from typing_extensions import Annotated, Literal, TypeAliasType, TypedDict, get_args, get_origin, is_typeddict
from ..aliases import AliasGenerator
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from ..config import ConfigDict, JsonDict, JsonEncoder
from ..errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation, PydanticUserError
from ..json_schema import JsonSchemaValue
from ..version import version_short
from ..warnings import PydanticDeprecatedSince20
from . import _core_utils, _decorators, _discriminated_union, _known_annotated_metadata, _typing_extra
from ._config import ConfigWrapper, ConfigWrapperStack
from ._core_metadata import CoreMetadataHandler, build_metadata_dict
from ._core_utils import (
from ._decorators import (
from ._fields import collect_dataclass_fields, get_type_hints_infer_globalns
from ._forward_ref import PydanticRecursiveRef
from ._generics import get_standard_typevars_map, has_instance_in_type, recursively_defined_type_refs, replace_types
from ._schema_generation_shared import (
from ._typing_extra import is_finalvar
from ._utils import lenient_issubclass
def _apply_annotations(self, source_type: Any, annotations: list[Any], transform_inner_schema: Callable[[CoreSchema], CoreSchema]=lambda x: x) -> CoreSchema:
    """Apply arguments from `Annotated` or from `FieldInfo` to a schema.

        This gets called by `GenerateSchema._annotated_schema` but differs from it in that it does
        not expect `source_type` to be an `Annotated` object, it expects it to be  the first argument of that
        (in other words, `GenerateSchema._annotated_schema` just unpacks `Annotated`, this process it).
        """
    annotations = list(_known_annotated_metadata.expand_grouped_metadata(annotations))
    res = self._get_prepare_pydantic_annotations_for_known_type(source_type, tuple(annotations))
    if res is not None:
        source_type, annotations = res
    pydantic_js_annotation_functions: list[GetJsonSchemaFunction] = []

    def inner_handler(obj: Any) -> CoreSchema:
        from_property = self._generate_schema_from_property(obj, obj)
        if from_property is None:
            schema = self._generate_schema(obj)
        else:
            schema = from_property
        metadata_js_function = _extract_get_pydantic_json_schema(obj, schema)
        if metadata_js_function is not None:
            metadata_schema = resolve_original_schema(schema, self.defs.definitions)
            if metadata_schema is not None:
                self._add_js_function(metadata_schema, metadata_js_function)
        return transform_inner_schema(schema)
    get_inner_schema = CallbackGetCoreSchemaHandler(inner_handler, self)
    for annotation in annotations:
        if annotation is None:
            continue
        get_inner_schema = self._get_wrapped_inner_schema(get_inner_schema, annotation, pydantic_js_annotation_functions)
    schema = get_inner_schema(source_type)
    if pydantic_js_annotation_functions:
        metadata = CoreMetadataHandler(schema).metadata
        metadata.setdefault('pydantic_js_annotation_functions', []).extend(pydantic_js_annotation_functions)
    return _add_custom_serialization_from_json_encoders(self._config_wrapper.json_encoders, source_type, schema)