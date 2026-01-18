from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MatchedPermissionsValue(_messages.Message):
    """The map from roles to their included permissions that match the
    permission query (i.e., a query containing `policy.role.permissions:`).
    Example: if query `policy.role.permissions:compute.disk.get` matches a
    policy binding that contains owner role, the matched_permissions will be
    `{"roles/owner": ["compute.disk.get"]}`. The roles can also be found in
    the returned `policy` bindings. Note that the map is populated only for
    requests with permission queries.

    Messages:
      AdditionalProperty: An additional property for a MatchedPermissionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        MatchedPermissionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MatchedPermissionsValue object.

      Fields:
        key: Name of the additional property.
        value: A Permissions attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Permissions', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)