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
@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class EncodedBytes:
    """A bytes type that is encoded and decoded using the specified encoder.

    `EncodedBytes` needs an encoder that implements `EncoderProtocol` to operate.

    ```py
    from typing_extensions import Annotated

    from pydantic import BaseModel, EncodedBytes, EncoderProtocol, ValidationError

    class MyEncoder(EncoderProtocol):
        @classmethod
        def decode(cls, data: bytes) -> bytes:
            if data == b'**undecodable**':
                raise ValueError('Cannot decode data')
            return data[13:]

        @classmethod
        def encode(cls, value: bytes) -> bytes:
            return b'**encoded**: ' + value

        @classmethod
        def get_json_format(cls) -> str:
            return 'my-encoder'

    MyEncodedBytes = Annotated[bytes, EncodedBytes(encoder=MyEncoder)]

    class Model(BaseModel):
        my_encoded_bytes: MyEncodedBytes

    # Initialize the model with encoded data
    m = Model(my_encoded_bytes=b'**encoded**: some bytes')

    # Access decoded value
    print(m.my_encoded_bytes)
    #> b'some bytes'

    # Serialize into the encoded form
    print(m.model_dump())
    #> {'my_encoded_bytes': b'**encoded**: some bytes'}

    # Validate encoded data
    try:
        Model(my_encoded_bytes=b'**undecodable**')
    except ValidationError as e:
        print(e)
        '''
        1 validation error for Model
        my_encoded_bytes
          Value error, Cannot decode data [type=value_error, input_value=b'**undecodable**', input_type=bytes]
        '''
    ```
    """
    encoder: type[EncoderProtocol]

    def __get_pydantic_json_schema__(self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format=self.encoder.get_json_format())
        return field_schema

    def __get_pydantic_core_schema__(self, source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(function=self.decode, schema=core_schema.bytes_schema(), serialization=core_schema.plain_serializer_function_ser_schema(function=self.encode))

    def decode(self, data: bytes, _: core_schema.ValidationInfo) -> bytes:
        """Decode the data using the specified encoder.

        Args:
            data: The data to decode.

        Returns:
            The decoded data.
        """
        return self.encoder.decode(data)

    def encode(self, value: bytes) -> bytes:
        """Encode the data using the specified encoder.

        Args:
            value: The data to encode.

        Returns:
            The encoded data.
        """
        return self.encoder.encode(value)

    def __hash__(self) -> int:
        return hash(self.encoder)