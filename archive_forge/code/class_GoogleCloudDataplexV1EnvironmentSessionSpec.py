from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EnvironmentSessionSpec(_messages.Message):
    """Configuration for sessions created for this environment.

  Fields:
    enableFastStartup: Optional. If True, this causes sessions to be pre-
      created and available for faster startup to enable interactive
      exploration use-cases. This defaults to False to avoid additional billed
      charges. These can only be set to True for the environment with name set
      to "default", and with default configuration.
    maxIdleDuration: Optional. The idle time configuration of the session. The
      session will be auto-terminated at the end of this period.
  """
    enableFastStartup = _messages.BooleanField(1)
    maxIdleDuration = _messages.StringField(2)