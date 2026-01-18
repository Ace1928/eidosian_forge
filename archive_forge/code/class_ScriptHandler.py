from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScriptHandler(_messages.Message):
    """Executes a script to handle the request that matches the URL pattern.

  Fields:
    scriptPath: Path to the script from the application root directory.
  """
    scriptPath = _messages.StringField(1)