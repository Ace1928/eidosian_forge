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
class CustomSearchPath(PatchedItem, TemporaryDirectory):
    """
    Context manager to temporarily customize ``$PATH`` (the executable search path).

    This class is a composition of the :class:`PatchedItem` and
    :class:`TemporaryDirectory` context managers.
    """

    def __init__(self, isolated=False):
        """
        Initialize a :class:`CustomSearchPath` object.

        :param isolated: :data:`True` to clear the original search path,
                         :data:`False` to add the temporary directory to the
                         start of the search path.
        """
        self.isolated_search_path = isolated
        PatchedItem.__init__(self, os.environ, 'PATH', self.current_search_path)
        TemporaryDirectory.__init__(self)

    def __enter__(self):
        """
        Activate the custom ``$PATH``.

        :returns: The pathname of the directory that has
                  been added to ``$PATH`` (a string).
        """
        directory = TemporaryDirectory.__enter__(self)
        self.patched_value = directory if self.isolated_search_path else os.pathsep.join([directory] + self.current_search_path.split(os.pathsep))
        PatchedItem.__enter__(self)
        return directory

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Deactivate the custom ``$PATH``."""
        super(CustomSearchPath, self).__exit__(exc_type, exc_value, traceback)

    @property
    def current_search_path(self):
        """The value of ``$PATH`` or :data:`os.defpath` (a string)."""
        return os.environ.get('PATH', os.defpath)