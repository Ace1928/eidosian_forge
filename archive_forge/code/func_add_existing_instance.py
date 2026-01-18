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
def add_existing_instance(cls, port):
    assert cls._is_port_taken(port), 'No Malmo mod utilizing the port specified.'
    instance = MinecraftInstance(port=port, existing=True, status_dir=None)
    cls._instance_pool.append(instance)
    cls.ninstances += 1
    return instance