import contextlib
import enum
import io
import os
import signal
import subprocess
import sys
import types
import typing
from typing import Any, Optional, Type, Dict, TextIO
from autopage import command
def _encoding(self) -> str:
    return getattr(self._out, 'encoding', 'ascii')