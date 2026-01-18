from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableFailure(_messages.Message):
    """Provides error messages for the failing services.

  Fields:
    errorMessage: An error message describing why the service could not be
      enabled.
    serviceId: The service id of a service that could not be enabled.
  """
    errorMessage = _messages.StringField(1)
    serviceId = _messages.StringField(2)