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
def _share_socket(sock: socket.SocketType) -> socket.SocketType:
    from socket import fromshare
    sock_data = sock.share(os.getpid())
    return fromshare(sock_data)