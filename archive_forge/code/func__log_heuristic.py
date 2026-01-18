import atexit
import functools
import locale
import logging
import multiprocessing
import os
import traceback
import pathlib
import Pyro4.core
import argparse
from enum import IntEnum
import shutil
import socket
import struct
import collections
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
import uuid
import psutil
import Pyro4
from random import Random
from minerl.env import comms
import minerl.utils.process_watcher
def _log_heuristic(self, msg):
    """
        Log the message, heuristically determine logging level based on the
        message content
        """
    if ('STDERR' in msg or 'ERROR' in msg or 'Exception' in msg or ('    at ' in msg) or msg.startswith('Error')) and (not 'connection closed, likely by peer' in msg):
        self._logger.error(msg)
    elif 'WARN' in msg:
        self._logger.warn(msg)
    elif 'LOGTOPY' in msg:
        self._logger.info(msg)
    else:
        self._logger.debug(msg)