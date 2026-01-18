from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
class _SubscriptionUpdateSetting(object):
    """Data container class for updating a subscription."""

    def __init__(self, field_name, value):
        self.field_name = field_name
        self.value = value