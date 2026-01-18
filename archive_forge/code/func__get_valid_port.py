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
@classmethod
def _get_valid_port(cls):
    malmo_base_port = cls._malmo_base_port
    port = cls.ninstances % 5000 + malmo_base_port
    port += 17 * os.getpid() % 3989
    while cls._is_port_taken(port) or cls._is_display_port_taken(port - malmo_base_port, cls.X11_DIR) or cls._port_in_instance_pool(port):
        port += 1
    return port