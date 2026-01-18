from abc import ABCMeta
from abc import abstractmethod
import argparse
import atexit
from collections import defaultdict
import errno
import logging
import mimetypes
import os
import shlex
import signal
import socket
import sys
import threading
import time
import urllib.parse
from absl import flags as absl_flags
from absl.flags import argparse_flags
from werkzeug import serving
from tensorboard import manager
from tensorboard import version
from tensorboard.backend import application
from tensorboard.backend.event_processing import data_ingester as local_ingester
from tensorboard.backend.event_processing import event_file_inspector as efi
from tensorboard.data import server_ingester
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
class TensorBoardSubcommand(metaclass=ABCMeta):
    """Experimental private API for defining subcommands to tensorboard(1)."""

    @abstractmethod
    def name(self):
        """Name of this subcommand, as specified on the command line.

        This must be unique across all subcommands.

        Returns:
          A string.
        """
        pass

    @abstractmethod
    def define_flags(self, parser):
        """Configure an argument parser for this subcommand.

        Flags whose names start with two underscores (e.g., `__foo`) are
        reserved for use by the runtime and must not be defined by
        subcommands.

        Args:
          parser: An `argparse.ArgumentParser` scoped to this subcommand,
            which this function should mutate.
        """
        pass

    @abstractmethod
    def run(self, flags):
        """Execute this subcommand with user-provided flags.

        Args:
          flags: An `argparse.Namespace` object with all defined flags.

        Returns:
          An `int` exit code, or `None` as an alias for `0`.
        """
        pass

    def help(self):
        """Short, one-line help text to display on `tensorboard --help`."""
        return None

    def description(self):
        """Description to display on `tensorboard SUBCOMMAND --help`."""
        return None