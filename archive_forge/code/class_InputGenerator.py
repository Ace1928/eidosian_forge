from __future__ import annotations
import abc
import copy
import os
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile
from monty.io import zopen
from monty.json import MSONable
class InputGenerator(MSONable):
    """
    InputGenerator classes serve as generators for Input objects. They contain
    settings or sets of instructions for how to create Input from a set of
    coordinates or a previous calculation directory.
    """

    @abc.abstractmethod
    def get_input_set(self, *args, **kwargs):
        """
        Generate an InputSet object. Typically the first argument to this method
        will be a Structure or other form of atomic coordinates.
        """
        raise NotImplementedError(f'get_input_set has not been implemented in {type(self).__name__}')