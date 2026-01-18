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
class SeedType(IntEnum):
    """The seed type for an instance manager.
    
    Values:
        0 - NONE: No seeding whatsoever.
        1 - CONSTANT: All envrionments have the same seed (the one specified 
            to the instance manager) (or alist of seeds , separated)
        2 - GENERATED: All environments have different seeds generated from a single 
            random generator with the seed specified to the InstanceManager.
        3 - SPECIFIED: Each instance is given a list of seeds. Specify this like
            1,2,3,4;848,432,643;888,888,888
            Each instance's seed list is separated by ; and each seed is separated by ,
    """
    NONE = 0
    CONSTANT = 1
    GENERATED = 2
    SPECIFIED = 3

    @classmethod
    def get_index(cls, type):
        return list(cls).index(type)