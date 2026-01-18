from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyTagsValue(_messages.Message):
    """Optional. The policy tags attached to this field, used for field-level
    access control. If not set, defaults to empty policy_tags.

    Fields:
      names: A list of policy tag resource names. For example,
        "projects/1/locations/eu/taxonomies/2/policyTags/3". At most 1 policy
        tag is currently allowed.
    """
    names = _messages.StringField(1, repeated=True)