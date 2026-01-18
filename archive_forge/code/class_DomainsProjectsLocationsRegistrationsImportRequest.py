from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsImportRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsImportRequest object.

  Fields:
    importDomainRequest: A ImportDomainRequest resource to be passed as the
      request body.
    parent: Required. The parent resource of the Registration. Must be in the
      format `projects/*/locations/*`.
  """
    importDomainRequest = _messages.MessageField('ImportDomainRequest', 1)
    parent = _messages.StringField(2, required=True)