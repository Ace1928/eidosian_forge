from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsRestoreRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsRestoreRequest object.

  Fields:
    name: Required. Resource name for the domain to which the backup belongs
    restoreDomainRequest: A RestoreDomainRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    restoreDomainRequest = _messages.MessageField('RestoreDomainRequest', 2)