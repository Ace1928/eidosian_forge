from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsGetRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsGetRequest object.

  Fields:
    name: Required. Name of the migration job resource to get.
  """
    name = _messages.StringField(1, required=True)