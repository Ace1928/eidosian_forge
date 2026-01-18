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
class SimdJsonModelSerializer:
    """
        JSON Encoder and Decoder using simdjson
        """
    parser = _parser

    @staticmethod
    def dumps(obj: typing.Dict[typing.Any, typing.Any], *args, default: typing.Dict[typing.Any, typing.Any]=None, cls: typing.Type[json.JSONEncoder]=ObjectModelEncoder, _fallback_method: typing.Optional[typing.Callable]=None, **kwargs) -> str:
        """
            Serializes a dict into a JSON string using the ObjectModelEncoder
            """
        try:
            return simdjson.dumps(obj, *args, default=default, cls=cls, **kwargs)
        except Exception as e:
            if _fallback_method is not None:
                return _fallback_method(obj, *args, default=default, **kwargs)
            raise e

    @staticmethod
    def loads(data: typing.Union[str, bytes], *args, object_hook: typing.Optional[typing.Callable]=object_model_deserializer, recursive: typing.Optional[bool]=True, _raw: typing.Optional[bool]=False, _fallback_method: typing.Optional[typing.Callable]=None, **kwargs) -> typing.Union[typing.Dict[typing.Any, typing.Any], typing.List[str], 'BaseModel', simdjson.Object, simdjson.Array]:
        """
            Loads a JSON string into a dict using the ObjectModelDecoder
            """
        try:
            value = _parser.parse(data, recursive)
            return value if _raw or not object_hook else object_hook(value)
        except Exception as e:
            if _fallback_method is not None:
                return _fallback_method(data, *args, **kwargs)
            raise e