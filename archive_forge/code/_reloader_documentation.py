from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
Spawn a new Python interpreter with the same arguments as the
        current one, but running the reloader thread.
        