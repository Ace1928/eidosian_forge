from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsResumeMigrationRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesMigratingVmsResumeMigrationRequest
  object.

  Fields:
    migratingVm: Required. The name of the MigratingVm.
    resumeMigrationRequest: A ResumeMigrationRequest resource to be passed as
      the request body.
  """
    migratingVm = _messages.StringField(1, required=True)
    resumeMigrationRequest = _messages.MessageField('ResumeMigrationRequest', 2)