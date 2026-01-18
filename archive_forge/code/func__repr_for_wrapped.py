import builtins
import codecs
import enum
import io
import json
import os
import types
import typing
from typing import (
import attr
def _repr_for_wrapped(self) -> str:
    return repr_for_fp(self._fp)