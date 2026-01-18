import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
class ExistsBaseDirectory(HasTraits):
    path = BaseDirectory(value=pathlib.Path(gettempdir()), exists=True)