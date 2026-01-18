from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def _NotificationCategoryEnumMapper(notification_category_enum_message):
    return arg_utils.ChoiceEnumMapper('--notification-categories', notification_category_enum_message)