import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class LifoSQLiteQueue(FifoSQLiteQueue):
    _sql_pop = 'SELECT id, item FROM queue ORDER BY id DESC LIMIT 1'