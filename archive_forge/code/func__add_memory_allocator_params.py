import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
def _add_memory_allocator_params(parser):
    group = parser.add_argument_group('Memory Allocator Parameters')
    group.add_argument('--enable-tcmalloc', '--enable_tcmalloc', action='store_true', default=False, help='Enable tcmalloc allocator')
    group.add_argument('--enable-jemalloc', '--enable_jemalloc', action='store_true', default=False, help='Enable jemalloc allocator')
    group.add_argument('--use-default-allocator', '--use_default_allocator', action='store_true', default=False, help='Use default memory allocator')