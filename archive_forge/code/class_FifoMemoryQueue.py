import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class FifoMemoryQueue:
    """In-memory FIFO queue, API compliant with FifoDiskQueue."""

    def __init__(self) -> None:
        self.q = deque()

    def push(self, obj: Any) -> None:
        self.q.append(obj)

    def pop(self) -> Optional[Any]:
        return self.q.popleft() if self.q else None

    def peek(self) -> Optional[Any]:
        return self.q[0] if self.q else None

    def close(self) -> None:
        pass

    def __len__(self):
        return len(self.q)