from __future__ import annotations
import asyncio
import logging
import os
import platform
import signal
import socket
import sys
import threading
import time
from email.utils import formatdate
from types import FrameType
from typing import TYPE_CHECKING, Sequence, Union
import click
from uvicorn.config import Config
class ServerState:
    """
    Shared servers state that is available between all protocol instances.
    """

    def __init__(self) -> None:
        self.total_requests = 0
        self.connections: set[Protocols] = set()
        self.tasks: set[asyncio.Task[None]] = set()
        self.default_headers: list[tuple[bytes, bytes]] = []