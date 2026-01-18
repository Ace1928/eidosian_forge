import typing
import warnings
import sys
from copy import deepcopy
from dataclasses import MISSING, is_dataclass, fields as dc_fields
from datetime import datetime
from decimal import Decimal
from uuid import UUID
from enum import Enum
from typing_inspect import is_union_type  # type: ignore
from marshmallow import fields, Schema, post_load  # type: ignore
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.core import (_is_supported_generic, _decode_dataclass,
from dataclasses_json.utils import (_is_collection, _is_optional,
class SchemaF(Schema, typing.Generic[A]):
    """Lift Schema into a type constructor"""

    def __init__(self, *args, **kwargs):
        """
            Raises exception because this class should not be inherited.
            This class is helper only.
            """
        super().__init__(*args, **kwargs)
        raise NotImplementedError()

    @typing.overload
    def dump(self, obj: typing.List[A], many: typing.Optional[bool]=None) -> typing.List[TEncoded]:
        pass

    @typing.overload
    def dump(self, obj: A, many: typing.Optional[bool]=None) -> TEncoded:
        pass

    def dump(self, obj: TOneOrMulti, many: typing.Optional[bool]=None) -> TOneOrMultiEncoded:
        pass

    @typing.overload
    def dumps(self, obj: typing.List[A], many: typing.Optional[bool]=None, *args, **kwargs) -> str:
        pass

    @typing.overload
    def dumps(self, obj: A, many: typing.Optional[bool]=None, *args, **kwargs) -> str:
        pass

    def dumps(self, obj: TOneOrMulti, many: typing.Optional[bool]=None, *args, **kwargs) -> str:
        pass

    @typing.overload
    def load(self, data: typing.List[TEncoded], many: bool=True, partial: typing.Optional[bool]=None, unknown: typing.Optional[str]=None) -> typing.List[A]:
        pass

    @typing.overload
    def load(self, data: TEncoded, many: None=None, partial: typing.Optional[bool]=None, unknown: typing.Optional[str]=None) -> A:
        pass

    def load(self, data: TOneOrMultiEncoded, many: typing.Optional[bool]=None, partial: typing.Optional[bool]=None, unknown: typing.Optional[str]=None) -> TOneOrMulti:
        pass

    @typing.overload
    def loads(self, json_data: JsonData, many: typing.Optional[bool]=True, partial: typing.Optional[bool]=None, unknown: typing.Optional[str]=None, **kwargs) -> typing.List[A]:
        pass

    def loads(self, json_data: JsonData, many: typing.Optional[bool]=None, partial: typing.Optional[bool]=None, unknown: typing.Optional[str]=None, **kwargs) -> TOneOrMulti:
        pass