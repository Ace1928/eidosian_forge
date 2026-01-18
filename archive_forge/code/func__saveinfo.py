import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
def _saveinfo(self, info: dict) -> None:
    with open(self._infopath(), 'w') as f:
        json.dump(info, f)