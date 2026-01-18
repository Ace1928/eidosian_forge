from __future__ import annotations
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Set, TYPE_CHECKING
class ArgParser(argparse.ArgumentParser):

    def print_help(self, file=None):
        """
        The default print_help method sends output to stdout, which isn't
        useful when run on the server. We have this raise the same exception
        as ``error`` so it gets handled in the same way (hopefully with
        help text going back to the user.)
        """
        raise ParserException('', help_text=self.format_help())

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        raise ParserException(message, help_text=self.format_help())