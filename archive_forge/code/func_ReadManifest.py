from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import with_statement
import re
import zipfile
def ReadManifest(jar_file_name):
    """Read and parse the manifest out of the given jar.

  Args:
    jar_file_name: the name of the jar from which the manifest is to be read.

  Returns:
    A parsed Manifest object, or None if the jar has no manifest.

  Raises:
    IOError: if the jar does not exist or cannot be read.
  """
    with zipfile.ZipFile(jar_file_name) as jar:
        try:
            manifest_string = jar.read(_MANIFEST_NAME).decode('utf-8')
        except KeyError:
            return None
        return _ParseManifest(manifest_string, jar_file_name)