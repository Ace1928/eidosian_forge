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
def ContactPrivacyEnumMapper(domains_messages):
    return arg_utils.ChoiceEnumMapper('--contact-privacy', _GetContactPrivacyEnum(domains_messages), custom_mappings={'PRIVATE_CONTACT_DATA': ('private-contact-data', "(DEPRECATED) Your contact info won't be available to the public. To help protect your info and prevent spam, a third party provides alternate (proxy) contact info for your domain in the public directory at no extra cost. They will forward received messages to you. The private-contact-data option is deprecated; See https://cloud.google.com/domains/docs/deprecations/feature-deprecations."), 'REDACTED_CONTACT_DATA': ('redacted-contact-data', 'Limited personal information will be available to the public. The actual information redacted depends on the domain. For more information see https://support.google.com/domains/answer/3251242.'), 'PUBLIC_CONTACT_DATA': ('public-contact-data', 'All the data from contact config is publicly available. To set this value, you must also pass the --notices flag with value public-contact-data-acknowledgement or agree to the notice interactively.')}, required=False, help_str='The contact privacy mode to use. Supported privacy modes depend on the domain.')