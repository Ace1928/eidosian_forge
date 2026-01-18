import time
import os
import sys
import hashlib
import gc
import shutil
import platform
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, Any
class _NodeCacheItem:

    def __init__(self, node, lines, change_time=None):
        self.node = node
        self.lines = lines
        if change_time is None:
            change_time = time.time()
        self.change_time = change_time
        self.last_used = change_time