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
class CustomAsyncRemoteMethod(Pyro4.core._AsyncRemoteMethod):

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        val = res.value
        if isinstance(val, Pyro4.Proxy):
            val._pyroAsync(asynchronous=True)
        return val