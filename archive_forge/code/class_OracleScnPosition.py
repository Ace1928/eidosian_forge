from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleScnPosition(_messages.Message):
    """Oracle SCN position

  Fields:
    scn: Required. SCN number from where Logs will be read
  """
    scn = _messages.IntegerField(1)