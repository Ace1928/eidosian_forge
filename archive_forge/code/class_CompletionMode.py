import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
class CompletionMode(Enum):
    """Enum for what type of tab completion to perform in cmd2.Cmd.read_input()"""
    NONE = 1
    COMMANDS = 2
    CUSTOM = 3