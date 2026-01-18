import click
import os
import shlex
import sys
from collections import defaultdict
from .exceptions import CommandLineParserError, ExitReplException
def _exit_internal():
    raise ExitReplException()