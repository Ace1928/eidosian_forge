from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def WriteBinaryFile(path, content):
    """Writes the given binary contents to the file at given path.

  Args:
      path (str): The file path to write to.
      content (str): The byte string to write.

  Raises:
      Error: If the file cannot be written.
  """
    if not path:
        return None
    try:
        files.WriteBinaryFileContents(path, content, private=True)
    except files.Error as e:
        raise exceptions.BadFileException('Unable to write file [{path}]: {e}'.format(path=path, e=e))