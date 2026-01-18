from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAccessPolicy(_messages.Message):
    """Policy describing who can access a service and any visibility labels on
  that service.

  Messages:
    VisibilityLabelAccessListsValue: ACLs for access to restricted parts of
      the service.  The map key is the visibility label that is being
      controlled.  Note that access to any label also implies access to the
      unrestricted surface.

  Fields:
    accessList: ACL for access to the unrestricted surface of the service.
    serviceName: The service protected by this policy.
    visibilityLabelAccessLists: ACLs for access to restricted parts of the
      service.  The map key is the visibility label that is being controlled.
      Note that access to any label also implies access to the unrestricted
      surface.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class VisibilityLabelAccessListsValue(_messages.Message):
        """ACLs for access to restricted parts of the service.  The map key is
    the visibility label that is being controlled.  Note that access to any
    label also implies access to the unrestricted surface.

    Messages:
      AdditionalProperty: An additional property for a
        VisibilityLabelAccessListsValue object.

    Fields:
      additionalProperties: Additional properties of type
        VisibilityLabelAccessListsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a VisibilityLabelAccessListsValue object.

      Fields:
        key: Name of the additional property.
        value: A ServiceAccessList attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ServiceAccessList', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accessList = _messages.MessageField('ServiceAccessList', 1)
    serviceName = _messages.StringField(2)
    visibilityLabelAccessLists = _messages.MessageField('VisibilityLabelAccessListsValue', 3)