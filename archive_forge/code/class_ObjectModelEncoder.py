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
class ObjectModelEncoder(json.JSONEncoder):
    """
    Object Model Encoder
    """

    def default(self, obj: typing.Any):
        with contextlib.suppress(Exception):
            return object_model_serializer(obj)
        return json.JSONEncoder.default(self, obj)