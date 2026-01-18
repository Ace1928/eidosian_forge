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
class ObjectDecoder(json.JSONDecoder):

    def __init__(self, *args, object_hook: typing.Optional[typing.Callable]=None, **kwargs):
        object_hook = object_hook or object_deserializer
        super().__init__(*args, object_hook=object_hook, **kwargs)