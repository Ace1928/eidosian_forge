import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
def _openchunk(self, number: int, mode: str='rb'):
    return open(os.path.join(self.path, 'q%05d' % number), mode)