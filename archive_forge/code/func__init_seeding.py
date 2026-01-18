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
def _init_seeding(cls, seed_type=int(SeedType.NONE), seeds=None):
    """Sets the seeding type of the Instance manager object.
        
        Args:
            seed_type (SeedType, optional): The seed type of the instancemanger.. Defaults to SeedType.NONE.
            seed (long, optional): The initial seed of the instance manager. Defaults to None.
        
        Raises:
            TypeError: If the SeedType specified does not fall within the SeedType.
        """
    seed_type = int(seed_type)
    if seed_type == SeedType.NONE:
        assert seeds is None, 'Seed type set to NONE, therefore seed cannot be set.'
    elif seed_type == SeedType.CONSTANT:
        assert seeds is not None, 'Seed set to constant seed, so seed must be specified.'
        cls._seed_generator = [int(x) for x in seeds.split(',') if x]
    elif seed_type == SeedType.GENERATED:
        assert seeds is not None, 'Seed set to generated seed, so initial seed must be specified.'
        cls._seed_generator = Random(int(seeds))
    elif seed_type == SeedType.SPECIFIED:
        cls._seed_generator = [[str(x) for x in s.split(',') if x] for s in seeds.split('-') if s]
    else:
        raise TypeError('Seed type {} not supported'.format(seed_type))
    cls._seed_type = seed_type