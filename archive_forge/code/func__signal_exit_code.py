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
def _signal_exit_code(signum: signal.Signals) -> int:
    """
    Return the exit code corresponding to a received signal.

    Conventionally, when a program exits due to a signal its exit code is 128
    plus the signal number.
    """
    return 128 + int(signum)