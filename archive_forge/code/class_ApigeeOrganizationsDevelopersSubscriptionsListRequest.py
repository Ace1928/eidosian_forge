from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersSubscriptionsListRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersSubscriptionsListRequest object.

  Fields:
    count: Number of API product subscriptions to return in the API call. Use
      with `startKey` to provide more targeted filtering. Defaults to 100. The
      maximum limit is 1000.
    parent: Required. Email address of the developer. Use the following
      structure in your request:
      `organizations/{org}/developers/{developer_email}`
    startKey: Name of the API product subscription from which to start
      displaying the list of subscriptions. If omitted, the list starts from
      the first item. For example, to view the API product subscriptions from
      51-150, set the value of `startKey` to the name of the 51st subscription
      and set the value of `count` to 100.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    parent = _messages.StringField(2, required=True)
    startKey = _messages.StringField(3)