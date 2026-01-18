from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.transfer.appliances import regions
def add_appliance_settings(parser, for_create_command=True):
    """Adds appliance flags for appliances orders create."""
    appliance_settings = parser.add_group(category='APPLIANCE')
    appliance_settings.add_argument('--model', choices=_APPLIANCE_MODELS, required=for_create_command, type=str.upper, help='Model of the appliance to order.')
    appliance_settings.add_argument('--display-name', help='A mutable, user-settable name for both the appliance and the order.')
    if for_create_command:
        appliance_settings.add_argument('--internet-enabled', action='store_true', help='Gives the option to put the appliance into online mode, allowing it to transfer data and/or remotely report progress to the cloud over the network. When disabled only offline transfers are possible.')
    appliance_settings.add_argument('--cmek', help='Customer-managed key which will add additional layer of security. By default data is encrypted with a Google-managed key.')
    appliance_settings.add_argument('--online-import', help='Destination for a online import, where data is loaded onto the appliance and automatically transferred to Cloud Storage whenever it has an internet connection. Should be in the form of `gs://my-bucket/path/to/folder/`.')
    appliance_settings.add_argument('--offline-import', help='Destination for a offline import, where data is loaded onto the appliance at the customer site and ingested at Google. Should be in the form of `gs://my-bucket/path/to/folder/`.')
    appliance_settings.add_argument('--offline-export', type=arg_parsers.ArgDict(spec={'source': str, 'manifest': str}), help=_OFFLINE_EXPORT_HELP)