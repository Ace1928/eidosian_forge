from __future__ import annotations
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from socket import socket
from types import FrameType
from typing import Callable, Iterator
import click
from uvicorn._subprocess import get_subprocess
from uvicorn.config import Config
def _display_path(path: Path) -> str:
    try:
        return f"'{path.relative_to(Path.cwd())}'"
    except ValueError:
        return f"'{path}'"