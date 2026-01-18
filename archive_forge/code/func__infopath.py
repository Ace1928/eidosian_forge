import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
def _infopath(self) -> str:
    return os.path.join(self.path, 'info.json')