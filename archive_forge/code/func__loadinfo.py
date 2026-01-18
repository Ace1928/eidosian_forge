import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
def _loadinfo(self, chunksize: int) -> dict:
    infopath = self._infopath()
    if os.path.exists(infopath):
        with open(infopath) as f:
            info = json.load(f)
    else:
        info = {'chunksize': chunksize, 'size': 0, 'tail': [0, 0, 0], 'head': [0, 0]}
    return info