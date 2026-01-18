from __future__ import annotations as _annotations
import base64
import dataclasses as _dataclasses
import re
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import (
from uuid import UUID
import annotated_types
from annotated_types import BaseMetadata, MaxLen, MinLen
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from typing_extensions import Annotated, Literal, Protocol, TypeAlias, TypeAliasType, deprecated
from ._internal import (
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .errors import PydanticUserError
from .json_schema import JsonSchemaValue
from .warnings import PydanticDeprecatedSince20
from pydantic import BaseModel, PositiveInt, ValidationError
from pydantic import BaseModel, NegativeInt, ValidationError
from pydantic import BaseModel, NonPositiveInt, ValidationError
from pydantic import BaseModel, NonNegativeInt, ValidationError
from pydantic import BaseModel, StrictInt, ValidationError
from pydantic import BaseModel, PositiveFloat, ValidationError
from pydantic import BaseModel, NegativeFloat, ValidationError
from pydantic import BaseModel, NonPositiveFloat, ValidationError
from pydantic import BaseModel, NonNegativeFloat, ValidationError
from pydantic import BaseModel, StrictFloat, ValidationError
from pydantic import BaseModel, FiniteFloat
import uuid
from pydantic import UUID1, BaseModel
import uuid
from pydantic import UUID3, BaseModel
import uuid
from pydantic import UUID4, BaseModel
import uuid
from pydantic import UUID5, BaseModel
from pathlib import Path
from pydantic import BaseModel, FilePath, ValidationError
from pathlib import Path
from pydantic import BaseModel, DirectoryPath, ValidationError
from pydantic import Base64Bytes, BaseModel, ValidationError
from pydantic import Base64Str, BaseModel, ValidationError
from pydantic import Base64UrlBytes, BaseModel
from pydantic import Base64UrlStr, BaseModel
class ImportString:
    """A type that can be used to import a type from a string.

        `ImportString` expects a string and loads the Python object importable at that dotted path.
        Attributes of modules may be separated from the module by `:` or `.`, e.g. if `'math:cos'` was provided,
        the resulting field value would be the function`cos`. If a `.` is used and both an attribute and submodule
        are present at the same path, the module will be preferred.

        On model instantiation, pointers will be evaluated and imported. There is
        some nuance to this behavior, demonstrated in the examples below.

        **Good behavior:**
        ```py
        from math import cos

        from pydantic import BaseModel, Field, ImportString, ValidationError


        class ImportThings(BaseModel):
            obj: ImportString


        # A string value will cause an automatic import
        my_cos = ImportThings(obj='math.cos')

        # You can use the imported function as you would expect
        cos_of_0 = my_cos.obj(0)
        assert cos_of_0 == 1


        # A string whose value cannot be imported will raise an error
        try:
            ImportThings(obj='foo.bar')
        except ValidationError as e:
            print(e)
            '''
            1 validation error for ImportThings
            obj
            Invalid python path: No module named 'foo.bar' [type=import_error, input_value='foo.bar', input_type=str]
            '''


        # Actual python objects can be assigned as well
        my_cos = ImportThings(obj=cos)
        my_cos_2 = ImportThings(obj='math.cos')
        my_cos_3 = ImportThings(obj='math:cos')
        assert my_cos == my_cos_2 == my_cos_3


        # You can set default field value either as Python object:
        class ImportThingsDefaultPyObj(BaseModel):
            obj: ImportString = math.cos


        # or as a string value (but only if used with `validate_default=True`)
        class ImportThingsDefaultString(BaseModel):
            obj: ImportString = Field(default='math.cos', validate_default=True)


        my_cos_default1 = ImportThingsDefaultPyObj()
        my_cos_default2 = ImportThingsDefaultString()
        assert my_cos_default1.obj == my_cos_default2.obj == math.cos


        # note: this will not work!
        class ImportThingsMissingValidateDefault(BaseModel):
            obj: ImportString = 'math.cos'

        my_cos_default3 = ImportThingsMissingValidateDefault()
        assert my_cos_default3.obj == 'math.cos'  # just string, not evaluated
        ```

        Serializing an `ImportString` type to json is also possible.

        ```py
        from pydantic import BaseModel, ImportString


        class ImportThings(BaseModel):
            obj: ImportString


        # Create an instance
        m = ImportThings(obj='math.cos')
        print(m)
        #> obj=<built-in function cos>
        print(m.model_dump_json())
        #> {"obj":"math.cos"}
        ```
        """

    @classmethod
    def __class_getitem__(cls, item: AnyType) -> AnyType:
        return Annotated[item, cls()]

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        serializer = core_schema.plain_serializer_function_ser_schema(cls._serialize, when_used='json')
        if cls is source:
            return core_schema.no_info_plain_validator_function(function=_validators.import_string, serialization=serializer)
        else:
            return core_schema.no_info_before_validator_function(function=_validators.import_string, schema=handler(source), serialization=serializer)

    @staticmethod
    def _serialize(v: Any) -> str:
        if isinstance(v, ModuleType):
            return v.__name__
        elif hasattr(v, '__module__') and hasattr(v, '__name__'):
            return f'{v.__module__}.{v.__name__}'
        else:
            return v

    def __repr__(self) -> str:
        return 'ImportString'