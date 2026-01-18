from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AcquireSsrsLeaseContext(_messages.Message):
    """Acquire SSRS lease context.

  Fields:
    duration: Lease duration needed for the SSRS setup.
    reportDatabase: The report database to be used for the SSRS setup.
    serviceLogin: The username to be used as the service login to connect to
      the report database for SSRS setup.
    setupLogin: The username to be used as the setup login to connect to the
      database server for SSRS setup.
  """
    duration = _messages.StringField(1)
    reportDatabase = _messages.StringField(2)
    serviceLogin = _messages.StringField(3)
    setupLogin = _messages.StringField(4)