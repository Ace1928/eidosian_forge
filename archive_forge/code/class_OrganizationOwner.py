from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrganizationOwner(_messages.Message):
    """The entity that owns an Organization. The lifetime of the Organization
  and all of its descendants are bound to the `OrganizationOwner`. If the
  `OrganizationOwner` is deleted, the Organization and all its descendants
  will be deleted.

  Fields:
    directoryCustomerId: The G Suite customer id used in the Directory API.
  """
    directoryCustomerId = _messages.StringField(1)