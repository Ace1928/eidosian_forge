from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def ReadFileOrStdin(path, max_bytes=None, is_binary=True):
    """Read data from the given file path or from stdin.

  This is similar to the cloudsdk built in ReadFromFileOrStdin, except that it
  limits the total size of the file and it returns None if given a None path.
  This makes the API in command surfaces a bit cleaner.

  Args:
      path (str): path to the file on disk or "-" for stdin
      max_bytes (int): maximum number of bytes
      is_binary (bool): if true, data will be read as binary

  Returns:
      result (str): result of reading the file
  """
    if not path:
        return None
    max_bytes = max_bytes or DEFAULT_MAX_BYTES
    try:
        data = console_io.ReadFromFileOrStdin(path, binary=is_binary)
        if len(data) > max_bytes:
            raise exceptions.BadFileException('The file [{path}] is larger than the maximum size of {max_bytes} bytes.'.format(path=path, max_bytes=max_bytes))
        return data
    except files.Error as e:
        raise exceptions.BadFileException('Failed to read file [{path}]: {e}'.format(path=path, e=e))