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
def _convert_schema(self, original_schema: core_schema.CoreSchema) -> core_schema.TaggedUnionSchema:
    if original_schema['type'] != 'union':
        original_schema = core_schema.union_schema([original_schema])
    tagged_union_choices = {}
    for i, choice in enumerate(original_schema['choices']):
        tag = None
        if isinstance(choice, tuple):
            choice, tag = choice
        metadata = choice.get('metadata')
        if metadata is not None:
            metadata_tag = metadata.get(_core_utils.TAGGED_UNION_TAG_KEY)
            if metadata_tag is not None:
                tag = metadata_tag
        if tag is None:
            raise PydanticUserError(f'`Tag` not provided for choice {choice} used with `Discriminator`', code='callable-discriminator-no-tag')
        tagged_union_choices[tag] = choice
    custom_error_type = self.custom_error_type
    if custom_error_type is None:
        custom_error_type = original_schema.get('custom_error_type')
    custom_error_message = self.custom_error_message
    if custom_error_message is None:
        custom_error_message = original_schema.get('custom_error_message')
    custom_error_context = self.custom_error_context
    if custom_error_context is None:
        custom_error_context = original_schema.get('custom_error_context')
    custom_error_type = original_schema.get('custom_error_type') if custom_error_type is None else custom_error_type
    return core_schema.tagged_union_schema(tagged_union_choices, self.discriminator, custom_error_type=custom_error_type, custom_error_message=custom_error_message, custom_error_context=custom_error_context, strict=original_schema.get('strict'), ref=original_schema.get('ref'), metadata=original_schema.get('metadata'), serialization=original_schema.get('serialization'))