from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AndroidTestLoop(_messages.Message):
    """Test Loops are tests that can be launched by the app itself, determining
  when to run by listening for an intent.
  """