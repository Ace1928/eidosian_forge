import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string
class MockedHomeDirectory(PatchedItem, TemporaryDirectory):
    """
    Context manager to temporarily change ``$HOME`` (the current user's profile directory).

    This class is a composition of the :class:`PatchedItem` and
    :class:`TemporaryDirectory` context managers.
    """

    def __init__(self):
        """Initialize a :class:`MockedHomeDirectory` object."""
        PatchedItem.__init__(self, os.environ, 'HOME', os.environ.get('HOME'))
        TemporaryDirectory.__init__(self)

    def __enter__(self):
        """
        Activate the custom ``$PATH``.

        :returns: The pathname of the directory that has
                  been added to ``$PATH`` (a string).
        """
        directory = TemporaryDirectory.__enter__(self)
        self.patched_value = directory
        PatchedItem.__enter__(self)
        return directory

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Deactivate the custom ``$HOME``."""
        super(MockedHomeDirectory, self).__exit__(exc_type, exc_value, traceback)