from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def ContactNoticeEnumMapper(domains_messages):
    return arg_utils.ChoiceEnumMapper('--notices', _GetContactNoticeEnum(domains_messages), custom_mappings={'PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT': ('public-contact-data-acknowledgement', 'By sending this notice you acknowledge that using public-contact-data contact privacy makes all the data from contact config publicly available.')}, required=False, help_str='Notices about special properties of contacts.')