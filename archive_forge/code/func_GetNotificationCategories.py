from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetNotificationCategories(args, notification_category_enum_message):
    if not args.notification_categories:
        return []
    return [arg_utils.ChoiceToEnum(category_choice, notification_category_enum_message) for category_choice in args.notification_categories]