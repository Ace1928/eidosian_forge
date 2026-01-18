from __future__ import annotations as _annotations
import typing
from typing import Any
import typing_extensions
class CoreMetadata(typing_extensions.TypedDict, total=False):
    """A `TypedDict` for holding the metadata dict of the schema.

    Attributes:
        pydantic_js_functions: List of JSON schema functions.
        pydantic_js_prefer_positional_arguments: Whether JSON schema generator will
            prefer positional over keyword arguments for an 'arguments' schema.
    """
    pydantic_js_functions: list[GetJsonSchemaFunction]
    pydantic_js_annotation_functions: list[GetJsonSchemaFunction]
    pydantic_js_prefer_positional_arguments: bool | None
    pydantic_typed_dict_cls: type[Any] | None