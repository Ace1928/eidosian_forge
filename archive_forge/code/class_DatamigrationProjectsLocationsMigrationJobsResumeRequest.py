from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsResumeRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsResumeRequest object.

  Fields:
    name: Name of the migration job resource to resume.
    resumeMigrationJobRequest: A ResumeMigrationJobRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    resumeMigrationJobRequest = _messages.MessageField('ResumeMigrationJobRequest', 2)