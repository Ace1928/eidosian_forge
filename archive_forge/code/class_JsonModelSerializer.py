import uuid
import json
import typing
import codecs
import hashlib
import datetime
import contextlib
import dataclasses
from enum import Enum
from .lazy import lazy_import, get_obj_class_name
class JsonModelSerializer:
    """
    Encoder and Decoder for Pydantic Models
    for optimal performance in deep serialization
    """

    @staticmethod
    def dumps(obj: typing.Dict[typing.Any, typing.Any], *args, default: typing.Dict[typing.Any, typing.Any]=None, cls: typing.Type[json.JSONEncoder]=ObjectModelEncoder, _fallback_method: typing.Optional[typing.Callable]=None, **kwargs) -> str:
        """
        Serializes a dict into a JSON string using the ObjectModelEncoder
        """
        try:
            return json.dumps(obj, *args, default=default, cls=cls, **kwargs)
        except Exception as e:
            if _fallback_method is not None:
                return _fallback_method(obj, *args, default=default, **kwargs)
            raise e

    @staticmethod
    def loads(data: typing.Union[str, bytes], *args, cls: typing.Type[json.JSONDecoder]=ObjectModelDecoder, _fallback_method: typing.Optional[typing.Callable]=None, **kwargs) -> typing.Union[typing.Dict[typing.Any, typing.Any], typing.List[str]]:
        """
        Loads a JSON string into a dict using the ObjectModelDecoder
        """
        try:
            return json.loads(data, *args, cls=cls, **kwargs)
        except json.JSONDecodeError as e:
            if _fallback_method is not None:
                return _fallback_method(data, *args, **kwargs)
            raise e