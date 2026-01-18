from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomaticUpdatePolicy(_messages.Message):
    """Security patches are applied automatically to the runtime without
  requiring the function to be redeployed.
  """