from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import import_util
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class Bak(base.Command):
    """Import data into a Cloud SQL instance from a BAK file."""
    detailed_help = {'DESCRIPTION': textwrap.dedent('          {command} imports data into a Cloud SQL instance from a BAK backup\n          file in Google Cloud Storage. You should use a full backup file with a\n          single backup set.\n\n          For detailed help on importing data into Cloud SQL, refer to this\n          guide: https://cloud.google.com/sql/docs/sqlserver/import-export/importing\n          '), 'EXAMPLES': textwrap.dedent('          To import data from the BAK file `my-bucket/my-export.bak` into the\n          database `my-database` in the Cloud SQL instance `my-instance`,\n          run:\n\n            $ {command} my-instance gs://my-bucket/my-export.bak --database=my-database\n\n          To import data from the encrypted BAK file `my-bucket/my-export.bak` into the database\n          `my-database` in the Cloud SQL instance `my-instance`, with the certificate\n          `gs://my-bucket/my-cert.crt`, private key `gs://my-bucket/my-key.key` and prompting for the\n          private key password,\n          run:\n\n            $ {command} my-instance gs://my-bucket/my-export.bak --database=my-database --cert-path=gs://my-bucket/my-cert.crt --pvk-path=gs://my-bucket/my-key.key --prompt-for-pvk-password\n          ')}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        import_util.AddBakImportFlags(parser, filetype='BAK file', gz_supported=False, user_supported=False)
        flags.AddDatabase(parser, flags.SQLSERVER_DATABASE_IMPORT_HELP_TEXT, required=True)
        flags.AddEncryptedBakFlags(parser)
        flags.AddBakImportStripedArgument(parser)
        flags.AddBakImportNoRecoveryArgument(parser)
        flags.AddBakImportRecoveryOnlyArgument(parser)
        flags.AddBakImportBakTypeArgument(parser)
        flags.AddBakImportStopAtArgument(parser)
        flags.AddBakImportStopAtMarkArgument(parser)

    def Run(self, args):
        """Runs the command to import into the Cloud SQL instance."""
        if args.prompt_for_pvk_password:
            args.pvk_password = console_io.PromptPassword('Private Key Password: ')
        client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
        return import_util.RunBakImportCommand(args, client)