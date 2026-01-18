import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
class LogLevel(enum.IntEnum):
    debug = logging.DEBUG
    info = logging.INFO
    warn = logging.WARN
    error = logging.ERROR
    fatal = logging.FATAL
    crit = logging.CRITICAL

    def __str__(self):
        return self.name