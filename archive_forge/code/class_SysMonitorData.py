from __future__ import annotations
import abc
import time
import atexit
import contextlib
from lazyops.libs import lazyload
from pydantic import BaseModel
from pydantic.types import ByteSize
from lazyops.utils import logger, Timer
from typing import Optional, List, Dict, Any, Union, Type, Set, TYPE_CHECKING
class SysMonitorData(BaseModel):
    """
    The System Monitor Data
    """
    cpu: Dict[str, float]
    memory: Dict[str, float]
    disk: Dict[str, float]
    network: Dict[str, float]
    process: Dict[str, float]
    system: Dict[str, float]