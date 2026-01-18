from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1ConsentPolicy(_messages.Message):
    """Represents a user's consent in terms of the resources that can be
  accessed and under what conditions.

  Fields:
    authorizationRule: Required. The request conditions to meet to grant
      access. In addition to any supported comparison operators, authorization
      rules may have `IN` operator as well as at most 10 logical operators
      that are limited to `AND` (`&&`), `OR` (`||`).
    resourceAttributes: The resources that this policy applies to. A resource
      is a match if it matches all the attributes listed here. If empty, this
      policy applies to all User data mappings for the given user.
  """
    authorizationRule = _messages.MessageField('Expr', 1)
    resourceAttributes = _messages.MessageField('Attribute', 2, repeated=True)