import os
import pathlib
import typing
import threading
import functools
import hashlib
from aiokeydb.types.compat import validator, root_validator, Field
from aiokeydb.types.compat import BaseSettings as _BaseSettings
from aiokeydb.types.compat import BaseModel as _BaseModel
from pydantic.networks import AnyUrl
class KeyDBDsn(AnyUrl):
    __slots__ = ()
    allowed_schemes = {'redis', 'rediss', 'keydb', 'keydbs'}
    host_required = False

    @staticmethod
    def get_default_parts(parts: Parts) -> Parts:
        return {'domain': '' if parts['ipv4'] or parts['ipv6'] else 'localhost', 'port': '6379', 'path': '/0'}