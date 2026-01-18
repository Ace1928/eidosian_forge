from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcpUserAccessBinding(_messages.Message):
    """Restricts access to Cloud Console and Google Cloud APIs for a set of
  users using Context-Aware Access. Next ID: 9

  Fields:
    accessLevels: Optional. Access level that a user must have to be granted
      access. Only one access level is supported, not multiple. This repeated
      field must have exactly one element. Example:
      "accessPolicies/9522/accessLevels/device_trusted"
    dryRunAccessLevels: Optional. Dry run access level that will be evaluated
      but will not be enforced. The access denial based on dry run policy will
      be logged. Only one access level is supported, not multiple. This list
      must have exactly one element. Example:
      "accessPolicies/9522/accessLevels/device_trusted"
    groupKey: Optional. Immutable. Google Group id whose members are subject
      to this binding's restrictions. See "id" in the [G Suite Directory API's
      Groups resource] (https://developers.google.com/admin-
      sdk/directory/v1/reference/groups#resource). If a group's email
      address/alias is changed, this resource will continue to point at the
      changed group. This field does not accept group email addresses or
      aliases. Example: "01d520gv4vjcrht"
    name: Immutable. Assigned by the server during creation. The last segment
      has an arbitrary length and has only URI unreserved characters (as
      defined by [RFC 3986 Section
      2.3](https://tools.ietf.org/html/rfc3986#section-2.3)). Should not be
      specified by the client during creation. Example:
      "organizations/256/gcpUserAccessBindings/b3-BhcX_Ud5N"
    restrictedClientApplications: Optional. A list of applications that are
      subject to this binding's restrictions. If the list is empty, the
      binding restrictions will universally apply to all applications. See
      go/caa-restricted-apps-control-plane.
  """
    accessLevels = _messages.StringField(1, repeated=True)
    dryRunAccessLevels = _messages.StringField(2, repeated=True)
    groupKey = _messages.StringField(3)
    name = _messages.StringField(4)
    restrictedClientApplications = _messages.MessageField('Application', 5, repeated=True)