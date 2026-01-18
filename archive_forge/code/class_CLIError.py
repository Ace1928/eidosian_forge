import argparse
import sys
import textwrap
from importlib import import_module
from ase import __version__
class CLIError(Exception):
    """Error for CLI commands.

    A subcommand may raise this.  The message will be forwarded to
    the error() method of the argument parser."""