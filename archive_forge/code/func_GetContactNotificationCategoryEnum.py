from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
def GetContactNotificationCategoryEnum(version=DEFAULT_API_VERSION):
    return GetContactMessage(version).NotificationCategorySubscriptionsValueListEntryValuesEnum