from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddNotificationCategoriesArg(parser, notification_category_enum_message, required=False, help_text='list of notification categories contact is subscribed to.'):
    """Adds the arg for specifying a list of notification categories to the parser."""
    parser.add_argument('--notification-categories', metavar='NOTIFICATION_CATEGORIES', type=arg_parsers.ArgList(choices=_NotificationCategoryEnumMapper(notification_category_enum_message).choices), help=help_text, required=required)