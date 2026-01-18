from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class ServiceAgent(base.Command):
    """Manage a project's Cloud Storage service agent, which is used to perform Cloud KMS operations."""
    detailed_help = {'DESCRIPTION': '\n      *{command}* displays the Cloud Storage service agent, which is used to\n      perform Cloud KMS operations against your a default or supplied project.\n      If the project does not yet have a service agent, *{command}* creates one.\n\n      ', 'EXAMPLES': "\n      To show the service agent for your default project:\n\n        $ {command}\n\n      To show the service account for ``my-project'':\n\n        $ {command} --project=my-project\n\n      To authorize your default project to use a Cloud KMS key:\n\n        $ {command} --authorize-cmek=projects/key-project/locations/us-east1/keyRings/key-ring/cryptoKeys/my-key\n      "}

    @staticmethod
    def Args(parser):
        parser.add_argument('--authorize-cmek', help=textwrap.dedent("            Adds appropriate encrypt/decrypt permissions to the specified Cloud\n            KMS key. This allows the Cloud Storage service agent to write and\n            read Cloud KMS-encrypted objects in buckets associated with the\n            service agent's project."))

    def Run(self, args):
        api = api_factory.get_api(storage_url.ProviderPrefix.GCS)
        service_agent = api.get_service_agent()
        if args.authorize_cmek:
            requests.AddCryptoKeyPermission(args.authorize_cmek, 'serviceAccount:' + service_agent)
            log.Print('Authorized project {} to encrypt and decrypt with key:\n{}'.format(properties.VALUES.core.project.Get(), args.authorize_cmek))
        else:
            log.Print(service_agent)