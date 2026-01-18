from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.firestore import flags
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateFirestoreAPIWithCmekConfig(CreateFirestoreAPI):
    """Create a Google Cloud Firestore database via Firestore API.

  ## EXAMPLES

  To create a Firestore Native database in `nam5`.

      $ {command} --location=nam5

  To create a Datastore Mode database in `us-east1`.

      $ {command} --location=us-east1 --type=datastore-mode

  To create a Datastore Mode database in `us-east1` with a databaseId `foo`.

      $ {command} --database=foo --location=us-east1 --type=datastore-mode

  To create a Firestore Native database in `nam5` with delete protection
  enabled.

      $ {command} --location=nam5 --delete-protection

  To create a Firestore Native database in `nam5` with Point In Time Recovery
  (PITR) enabled.

      $ {command} --location=nam5 --enable-pitr
  """

    def DatabaseCmekConfig(self, args):
        if args.kms_key_name is not None:
            return api_utils.GetMessages().GoogleFirestoreAdminV1CmekConfig(kmsKeyName=args.kms_key_name)
        return api_utils.GetMessages().GoogleFirestoreAdminV1CmekConfig()

    @classmethod
    def Args(cls, parser):
        super(CreateFirestoreAPIWithCmekConfig, cls).Args(parser)
        parser.add_argument('--kms-key-name', help="The resource ID of a Cloud KMS key. If set, the database created will\n        be a Customer-managed Encryption Key (CMEK) database encrypted with\n        this key. This feature is allowlist only in initial launch.\n\n        Only the key in the same location as this database is allowed to be\n        used for encryption.\n\n        For Firestore's nam5 multi-region, this corresponds to Cloud KMS\n        location us. For Firestore's eur3 multi-region, this corresponds to\n        Cloud KMS location europe. See https://cloud.google.com/kms/docs/locations.\n\n        This value should be the KMS key resource ID in the format of\n        `projects/{project_id}/locations/{kms_location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}`.\n        How to retrive this resource ID is listed at https://cloud.google.com/kms/docs/getting-resource-ids#getting_the_id_for_a_key_and_version.\n        ", type=str, default=None)