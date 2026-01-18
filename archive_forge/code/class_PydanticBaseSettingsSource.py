from __future__ import annotations as _annotations
import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple, Union, cast
from dotenv import dotenv_values
from pydantic import AliasChoices, AliasPath, BaseModel, Json
from pydantic._internal._typing_extra import WithArgsTypes, origin_is_union
from pydantic._internal._utils import deep_update, is_model_class, lenient_issubclass
from pydantic.fields import FieldInfo
from typing_extensions import get_args, get_origin
from pydantic_settings.utils import path_type_label
class PydanticBaseSettingsSource(ABC):
    """
    Abstract base class for settings sources, every settings source classes should inherit from it.
    """

    def __init__(self, settings_cls: type[BaseSettings]):
        self.settings_cls = settings_cls
        self.config = settings_cls.model_config

    @abstractmethod
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """
        Gets the value, the key for model creation, and a flag to determine whether value is complex.

        This is an abstract method that should be overridden in every settings source classes.

        Args:
            field: The field.
            field_name: The field name.

        Returns:
            A tuple contains the key, value and a flag to determine whether value is complex.
        """
        pass

    def field_is_complex(self, field: FieldInfo) -> bool:
        """
        Checks whether a field is complex, in which case it will attempt to be parsed as JSON.

        Args:
            field: The field.

        Returns:
            Whether the field is complex.
        """
        return _annotation_is_complex(field.annotation, field.metadata)

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        """
        Prepares the value of a field.

        Args:
            field_name: The field name.
            field: The field.
            value: The value of the field that has to be prepared.
            value_is_complex: A flag to determine whether value is complex.

        Returns:
            The prepared value.
        """
        if value is not None and (self.field_is_complex(field) or value_is_complex):
            return self.decode_complex_value(field_name, field, value)
        return value

    def decode_complex_value(self, field_name: str, field: FieldInfo, value: Any) -> Any:
        """
        Decode the value for a complex field

        Args:
            field_name: The field name.
            field: The field.
            value: The value of the field that has to be prepared.

        Returns:
            The decoded value for further preparation
        """
        return json.loads(value)

    @abstractmethod
    def __call__(self) -> dict[str, Any]:
        pass