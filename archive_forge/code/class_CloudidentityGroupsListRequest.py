from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsListRequest(_messages.Message):
    """A CloudidentityGroupsListRequest object.

  Enums:
    ViewValueValuesEnum: The level of detail to be returned. If unspecified,
      defaults to `View.BASIC`.

  Fields:
    pageSize: The maximum number of results to return. Note that the number of
      results returned may be less than this value even if there are more
      available results. To fetch all results, clients must continue calling
      this method repeatedly until the response no longer contains a
      `next_page_token`. If unspecified, defaults to 200 for `View.BASIC` and
      to 50 for `View.FULL`. Must not be greater than 1000 for `View.BASIC` or
      500 for `View.FULL`.
    pageToken: The `next_page_token` value returned from a previous list
      request, if any.
    parent: Required. The parent resource under which to list all `Group`
      resources. Must be of the form `identitysources/{identity_source}` for
      external- identity-mapped groups or `customers/{customer_id}` for Google
      Groups. The `customer_id` must begin with "C" (for example,
      'C046psxkn'). [Find your customer ID.]
      (https://support.google.com/cloudidentity/answer/10070793)
    view: The level of detail to be returned. If unspecified, defaults to
      `View.BASIC`.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The level of detail to be returned. If unspecified, defaults to
    `View.BASIC`.

    Values:
      VIEW_UNSPECIFIED: Default. Should not be used.
      BASIC: Only basic resource information is returned.
      FULL: All resource information is returned.
    """
        VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)
    view = _messages.EnumField('ViewValueValuesEnum', 4)