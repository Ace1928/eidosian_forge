import unittest
import contextlib
import pathlib
import importlib_resources as resources
from .. import abc
from ..abc import TraversableResources, ResourceReader
from . import util
from .compat.py39 import os_helper
class SimpleLoader:
    """
    A simple loader that only implements a resource reader.
    """

    def __init__(self, reader: ResourceReader):
        self.reader = reader

    def get_resource_reader(self, package):
        return self.reader