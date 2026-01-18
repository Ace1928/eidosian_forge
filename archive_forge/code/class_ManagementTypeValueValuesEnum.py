from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementTypeValueValuesEnum(_messages.Enum):
    """[Output Only] The resource that configures and manages this interface.
    - MANAGED_BY_USER is the default value and can be managed directly by
    users. - MANAGED_BY_ATTACHMENT is an interface that is configured and
    managed by Cloud Interconnect, specifically, by an InterconnectAttachment
    of type PARTNER. Google automatically creates, updates, and deletes this
    type of interface when the PARTNER InterconnectAttachment is created,
    updated, or deleted.

    Values:
      MANAGED_BY_ATTACHMENT: The interface is automatically created for
        PARTNER type InterconnectAttachment, Google will automatically
        create/update/delete this interface when the PARTNER
        InterconnectAttachment is created/provisioned/deleted. This type of
        interface cannot be manually managed by user.
      MANAGED_BY_USER: Default value, the interface is manually created and
        managed by user.
    """
    MANAGED_BY_ATTACHMENT = 0
    MANAGED_BY_USER = 1