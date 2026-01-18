from __future__ import annotations
import copy
import datetime
import itertools
from typing import Any, Generic, Mapping, Optional
from bson.objectid import ObjectId
from pymongo import common
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import ClusterTime, _DocumentType
class HelloCompat:
    CMD = 'hello'
    LEGACY_CMD = 'ismaster'
    PRIMARY = 'isWritablePrimary'
    LEGACY_PRIMARY = 'ismaster'
    LEGACY_ERROR = 'not master'