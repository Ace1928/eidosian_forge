from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
class FileConfigLoader(ConfigLoader):
    """A base class for file based configurations.

    As we add more file based config loaders, the common logic should go
    here.
    """

    def __init__(self, filename: str, path: str | None=None, **kw: t.Any) -> None:
        """Build a config loader for a filename and path.

        Parameters
        ----------
        filename : str
            The file name of the config file.
        path : str, list, tuple
            The path to search for the config file on, or a sequence of
            paths to try in order.
        """
        super().__init__(**kw)
        self.filename = filename
        self.path = path
        self.full_filename = ''

    def _find_file(self) -> None:
        """Try to find the file by searching the paths."""
        self.full_filename = filefind(self.filename, self.path)