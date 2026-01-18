import abc
import dataclasses
import functools
import inspect
import sys
from dataclasses import Field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union, Type, get_type_hints
from enum import Enum
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.utils import CatchAllVar
@staticmethod
def _separate_defined_undefined_kvs(cls, kvs: Dict) -> Tuple[KnownParameters, UnknownParameters]:
    """
        Returns a 2 dictionaries: defined and undefined parameters
        """
    class_fields = fields(cls)
    field_names = [field.name for field in class_fields]
    unknown_given_parameters = {k: v for k, v in kvs.items() if k not in field_names}
    known_given_parameters = {k: v for k, v in kvs.items() if k in field_names}
    return (known_given_parameters, unknown_given_parameters)