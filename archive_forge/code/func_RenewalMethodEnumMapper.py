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
def RenewalMethodEnumMapper(domains_messages):
    return arg_utils.ChoiceEnumMapper('--preferred-renewal-method', _GetRenewalMethodEnum(domains_messages), custom_mappings={'AUTOMATIC_RENEWAL': ('automatic-renewal', 'The domain is automatically renewed each year.'), 'RENEWAL_DISABLED': ('renewal-disabled', "The domain won't be renewed and will expire at its expiration time.")}, required=False, help_str='Preferred Renewal Method of a registration. It defines how the registration should be renewed. The actual Renewal Method can be set to renewal-disabled in case of e.g. problems with the Billing Account or reporeted domain abuse.')