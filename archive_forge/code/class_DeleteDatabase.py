from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class DeleteDatabase(base.Command):
    """Delete a Google Cloud Firestore database.

  ## EXAMPLES

  To delete a Firestore database test.

      $ {command} --database=test

  To delete the Firestore (default) database.

      $ {command} --database=(default)

  To delete a Firestore database test providing etag.

      $ {command} --database=test --etag=etag
  """

    def Run(self, args):
        project = properties.VALUES.core.project.Get(required=True)
        console_io.PromptContinue(message="The database 'projects/{}/databases/{}' will be deleted.".format(project, args.database), cancel_on_no=True)
        return databases.DeleteDatabase(project, args.database, args.etag)

    @staticmethod
    def Args(parser):
        parser.add_argument('--database', help='The database to operate on.', type=str, required=True)
        parser.add_argument('--etag', help='The current etag of the Database. If an etag is provided and does not match the current etag of the database, deletion will be blocked and a FAILED_PRECONDITION error will be returned.', type=str)