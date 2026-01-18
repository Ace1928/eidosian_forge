from __future__ import annotations
import logging # isort:skip
import argparse
import sys
from abc import abstractmethod
from os.path import splitext
from ...document import Document
from ..subcommand import (
from ..util import build_single_handler_applications, die
 Subclasses must override this method to return the contents of the output file for the given doc.
        subclassed methods return different types:
        str: html, json
        bytes: SVG, png

        Raises:
            NotImplementedError

        