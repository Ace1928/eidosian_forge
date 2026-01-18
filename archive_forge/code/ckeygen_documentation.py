from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage

    Persist a SSH key on local filesystem.

    @param key: Key which is persisted on local filesystem.

    @param options:

    @param inputCollector: Dependency injection for testing.
    