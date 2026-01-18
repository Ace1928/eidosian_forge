import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
@enum.unique
class UpdateMethod(enum.IntEnum):
    PUT = 1
    POST = 2
    PATCH = 3