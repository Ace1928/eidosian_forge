from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import os
import tempfile
class _WindowsNamedTempFile(object):
    """Wrapper around named temporary file for Windows.

  NamedTemporaryFiles cannot be read by other processes on windows because
  only one process can open a file at a time. This file will be unlinked
  at the end of the context.
  """

    def __init__(self, *args, **kwargs):
        self._requested_delete = kwargs.get('delete', True)
        self._args = args
        self._kwargs = kwargs.copy()
        self._kwargs['delete'] = False
        self._f = None

    def __enter__(self):
        self._f = tempfile.NamedTemporaryFile(*self._args, **self._kwargs)
        return self._f

    def __exit__(self, exc_type, exc_value, tb):
        if self._requested_delete and self._f:
            try:
                os.unlink(self._f.name)
            except OSError:
                pass