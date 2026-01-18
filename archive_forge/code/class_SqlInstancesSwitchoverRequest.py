from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesSwitchoverRequest(_messages.Message):
    """A SqlInstancesSwitchoverRequest object.

  Fields:
    dbTimeout: Optional. (MySQL only) Cloud SQL instance operations timeout,
      which is a sum of all database operations. Default value is 10 minutes
      and can be modified to a maximum value of 24 hours.
    instance: Cloud SQL read replica instance name.
    project: ID of the project that contains the replica.
  """
    dbTimeout = _messages.StringField(1)
    instance = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)