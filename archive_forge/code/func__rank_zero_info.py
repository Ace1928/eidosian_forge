import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Union
from lightning_fabric.utilities.cloud_io import get_filesystem
def _rank_zero_info(self, *args: Any, **kwargs: Any) -> None:
    if self._local_rank in (None, 0):
        log.info(*args, **kwargs)