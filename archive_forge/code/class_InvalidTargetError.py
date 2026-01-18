from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
class InvalidTargetError(Exception):
    """
    Invalid Target Error
    """

    def __init__(self, converter: 'BaseConverter', target: str):
        """
        Initialize the Invalid Target Error
        """
        self.converter = converter
        self.target = target
        super().__init__(f'[{self.converter.name}] Invalid conversion target: {target} (supported: {self.converter.targets})')