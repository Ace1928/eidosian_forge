import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class BaseQueue(metaclass=_BaseQueueMeta):

    @abstractmethod
    def push(self, obj: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def pop(self) -> Optional[Any]:
        raise NotImplementedError()

    @abstractmethod
    def peek(self) -> Optional[Any]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def close(self) -> None:
        pass