from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsSearchRequest(_messages.Message):
    """A CloudidentityGroupsSearchRequest object.

  Enums:
    ViewValueValuesEnum: The level of detail to be returned. If unspecified,
      defaults to `View.BASIC`.

  Fields:
    pageSize: The maximum number of results to return. Note that the number of
      results returned may be less than this value even if there are more
      available results. To fetch all results, clients must continue calling
      this method repeatedly until the response no longer contains a
      `next_page_token`. If unspecified, defaults to 200 for `GroupView.BASIC`
      and 50 for `GroupView.FULL`. Must not be greater than 1000 for
      `GroupView.BASIC` or 500 for `GroupView.FULL`.
    pageToken: The `next_page_token` value returned from a previous search
      request, if any.
    query: Required. The search query. * Must be specified in [Common
      Expression Language](https://opensource.google/projects/cel). * Must
      contain equality operators on the parent, e.g. `parent ==
      'customers/{customer_id}'`. The `customer_id` must begin with "C" (for
      example, 'C046psxkn'). [Find your customer ID.]
      (https://support.google.com/cloudidentity/answer/10070793) * Can contain
      optional inclusion operators on `labels` such as
      `'cloudidentity.googleapis.com/groups.discussion_forum' in labels`). *
      Can contain an optional equality operator on `domain_name`. e.g.
      `domain_name == 'examplepetstore.com'` * Can contain optional
      `startsWith/contains/equality` operators on `group_key`, e.g.
      `group_key.startsWith('dev')`, `group_key.contains('dev'), group_key ==
      'dev@examplepetstore.com'` * Can contain optional
      `startsWith/contains/equality` operators on `display_name`, such as
      `display_name.startsWith('dev')` , `display_name.contains('dev')`,
      `display_name == 'dev'`
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
    query = _messages.StringField(3)
    view = _messages.EnumField('ViewValueValuesEnum', 4)