from __future__ import absolute_import
import re
def ConvertEntrypoint(entrypoint):
    """Converts the raw entrypoint to a nested shell value.

  In the YAML file, the user specifies an entrypoint value. However, the version
  resource expects it to be nested under a 'shell' key. In addition, Zeus
  always prepends 'exec' to the value provided, so we remove it here as it is
  sometimes added client-side by the validation library.

  Args:
    entrypoint: string, entrypoint value.

  Returns:
    Dict containing entrypoint.
  """
    if entrypoint is None:
        entrypoint = ''
    if entrypoint.startswith('exec '):
        entrypoint = entrypoint[len('exec '):]
    return {'shell': entrypoint}