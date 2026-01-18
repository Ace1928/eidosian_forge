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

        A signal handler that is registered with the parent process.
        