from __future__ import annotations
from typing import ClassVar, List
import param
from ..config import config as pn_config
from ..io.resources import CDN_DIST, bundled_files
from ..reactive import ReactiveHTML
from ..util import classproperty
from .base import ListLike
@classproperty
def __css__(cls):
    return bundled_files(cls, 'css')