from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepositoryConfig(_messages.Message):
    """Configuration for dependency repositories

  Fields:
    pypiRepositoryConfig: Optional. Configuration for PyPi repository.
  """
    pypiRepositoryConfig = _messages.MessageField('PyPiRepositoryConfig', 1)