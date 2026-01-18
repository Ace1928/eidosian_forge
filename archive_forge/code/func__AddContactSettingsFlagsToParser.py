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
def _AddContactSettingsFlagsToParser(parser, mutation_op):
    """Get flags for providing Contact settings.

  Args:
    parser: argparse parser to which to add these flags.
    mutation_op: operation for which we're adding flags.
  """
    help_text = "    A YAML file containing the contact data for the domain's three contacts:\n    registrant, admin, and technical.\n\n    The file can either specify a single set of contact data with label\n    'allContacts', or three separate sets of contact data with labels\n    'adminContact' and 'technicalContact'.\n    {}\n    Each contact data must contain values for all required fields: email,\n    phoneNumber and postalAddress in google.type.PostalAddress format.\n\n    For more guidance on how to specify postalAddress, please see:\n    https://support.google.com/business/answer/6397478\n\n    Examples of file contents:\n\n    ```\n    allContacts:\n      email: 'example@example.com'\n      phoneNumber: '+1.8005550123'\n      postalAddress:\n        regionCode: 'US'\n        postalCode: '94043'\n        administrativeArea: 'CA'\n        locality: 'Mountain View'\n        addressLines: ['1600 Amphitheatre Pkwy']\n        recipients: ['Jane Doe']\n    ```\n    {}\n    ```\n    registrantContact:\n      email: 'registrant@example.com'\n      phoneNumber: '+1.8005550123'\n      postalAddress:\n        regionCode: 'US'\n        postalCode: '94043'\n        administrativeArea: 'CA'\n        locality: 'Mountain View'\n        addressLines: ['1600 Amphitheatre Pkwy']\n        recipients: ['Registrant Jane Doe']\n    adminContact:\n      email: 'admin@example.com'\n      phoneNumber: '+1.8005550123'\n      postalAddress:\n        regionCode: 'US'\n        postalCode: '94043'\n        administrativeArea: 'CA'\n        locality: 'Mountain View'\n        addressLines: ['1600 Amphitheatre Pkwy']\n        recipients: ['Admin Jane Doe']\n    technicalContact:\n      email: 'technical@example.com'\n      phoneNumber: '+1.8005550123'\n      postalAddress:\n        regionCode: 'US'\n        postalCode: '94043'\n        administrativeArea: 'CA'\n        locality: 'Mountain View'\n        addressLines: ['1600 Amphitheatre Pkwy']\n        recipients: ['Technic Jane Doe']\n    ```\n    "
    if mutation_op == MutationOp.UPDATE:
        help_text = help_text.format("\n    If 'registrantContact', 'adminContact' or 'technicalContact' labels are used\n    then only the specified contacts are updated.\n    ", "\n    ```\n    adminContact:\n      email: 'admin@example.com'\n      phoneNumber: '+1.8005550123'\n      postalAddress:\n        regionCode: 'US'\n        postalCode: '94043'\n        administrativeArea: 'CA'\n        locality: 'Mountain View'\n        addressLines: ['1600 Amphitheatre Pkwy']\n        recipients: ['Admin Jane Doe']\n    ```\n        ")
    else:
        help_text = help_text.format('', '')
    base.Argument('--contact-data-from-file', help=help_text, metavar='CONTACT_DATA_FILE_NAME', category=base.COMMONLY_USED_FLAGS).AddToParser(parser)

    def _ChoiceValueType(value):
        """Copy of base._ChoiceValueType."""
        return value.replace('_', '-').lower()
    messages = apis.GetMessagesModule('domains', API_VERSION_FOR_FLAGS)
    base.Argument('--contact-privacy', choices=ContactPrivacyEnumMapper(messages).choices, type=_ChoiceValueType, help='The contact privacy mode to use. Supported privacy modes depend on the domain.', required=False, category=base.COMMONLY_USED_FLAGS, action=actions.DeprecationAction('--contact-privacy=private-contact-data', show_message=lambda choice: choice == 'private-contact-data', show_add_help=False, warn='The {flag_name} option is deprecated; See https://cloud.google.com/domains/docs/deprecations/feature-deprecations.', removed=False)).AddToParser(parser)