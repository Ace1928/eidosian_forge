from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PyPiRepositoryConfig(_messages.Message):
    """Configuration for PyPi repository

  Fields:
    pypiRepository: Optional. PyPi repository address
  """
    pypiRepository = _messages.StringField(1)