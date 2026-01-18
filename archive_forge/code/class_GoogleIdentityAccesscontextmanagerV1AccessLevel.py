from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1AccessLevel(_messages.Message):
    """An `AccessLevel` is a label that can be applied to requests to Google
  Cloud services, along with a list of requirements necessary for the label to
  be applied.

  Fields:
    basic: A `BasicLevel` composed of `Conditions`.
    custom: A `CustomLevel` written in the Common Expression Language.
    description: Description of the `AccessLevel` and its use. Does not affect
      behavior.
    name: Resource name for the `AccessLevel`. Format:
      `accessPolicies/{access_policy}/accessLevels/{access_level}`. The
      `access_level` component must begin with a letter, followed by
      alphanumeric characters or `_`. Its maximum length is 50 characters.
      After you create an `AccessLevel`, you cannot change its `name`.
    title: Human readable title. Must be unique within the Policy.
  """
    basic = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1BasicLevel', 1)
    custom = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1CustomLevel', 2)
    description = _messages.StringField(3)
    name = _messages.StringField(4)
    title = _messages.StringField(5)