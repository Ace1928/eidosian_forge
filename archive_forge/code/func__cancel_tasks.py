import asyncio
import logging
import os
import socket
import sys
import warnings
from argparse import ArgumentParser
from collections.abc import Iterable
from contextlib import suppress
from functools import partial
from importlib import import_module
from typing import (
from weakref import WeakSet
from .abc import AbstractAccessLogger
from .helpers import AppKey as AppKey
from .log import access_logger
from .typedefs import PathLike
from .web_app import Application as Application, CleanupError as CleanupError
from .web_exceptions import (
from .web_fileresponse import FileResponse as FileResponse
from .web_log import AccessLogger
from .web_middlewares import (
from .web_protocol import (
from .web_request import (
from .web_response import (
from .web_routedef import (
from .web_runner import (
from .web_server import Server as Server
from .web_urldispatcher import (
from .web_ws import (
def _cancel_tasks(to_cancel: Set['asyncio.Task[Any]'], loop: asyncio.AbstractEventLoop) -> None:
    if not to_cancel:
        return
    for task in to_cancel:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))
    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler({'message': 'unhandled exception during asyncio.run() shutdown', 'exception': task.exception(), 'task': task})