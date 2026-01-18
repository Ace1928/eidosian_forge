import asyncio
import functools
import os
import re
import signal
import sys
import typing as t
import uuid
import warnings
from asyncio.futures import Future
from concurrent.futures import Future as CFuture
from contextlib import contextmanager
from enum import Enum
import zmq
from jupyter_core.utils import run_sync
from traitlets import (
from traitlets.utils.importstring import import_item
from . import kernelspec
from .asynchronous import AsyncKernelClient
from .blocking import BlockingKernelClient
from .client import KernelClient
from .connect import ConnectionFileMixin
from .managerabc import KernelManagerABC
from .provisioning import KernelProvisionerBase
from .provisioning import KernelProvisionerFactory as KPF  # noqa
class _ShutdownStatus(Enum):
    """

    This is so far used only for testing in order to track the internal state of
    the shutdown logic, and verifying which path is taken for which
    missbehavior.

    """
    Unset = None
    ShutdownRequest = 'ShutdownRequest'
    SigtermRequest = 'SigtermRequest'
    SigkillRequest = 'SigkillRequest'