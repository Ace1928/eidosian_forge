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
class CustomCompletionSettings:
    """Used by cmd2.Cmd.complete() to tab complete strings other than command arguments"""

    def __init__(self, parser: argparse.ArgumentParser, *, preserve_quotes: bool=False) -> None:
        """
        Initializer

        :param parser: arg parser defining format of string being tab completed
        :param preserve_quotes: if True, then quoted tokens will keep their quotes when processed by
                                ArgparseCompleter. This is helpful in cases when you're tab completing
                                flag-like tokens (e.g. -o, --option) and you don't want them to be
                                treated as argparse flags when quoted. Set this to True if you plan
                                on passing the string to argparse with the tokens still quoted.
        """
        self.parser = parser
        self.preserve_quotes = preserve_quotes