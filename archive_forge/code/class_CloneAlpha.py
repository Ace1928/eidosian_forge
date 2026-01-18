from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.source import git
from googlecloudsdk.api_lib.source import sourcerepo
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store as c_store
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class CloneAlpha(CloneGA):
    """Clone a cloud source repository.

  This command clones a git repository for the currently active
  Google Cloud Platform project into the specified directory or into
  the current directory if no target directory is specified.  This command
  gives an error if the cloud source repository is a mirror.

  The clone operation configures the local clone to use your gcloud
  credentials to authenticate future git operations.

  ## EXAMPLES

  The example commands below show a sample workflow.

    $ gcloud init
    $ {command} REPOSITORY_NAME DIRECTORY_NAME
    $ cd DIRECTORY_NAME
    ... create/edit files and create one or more commits ...
    $ git push origin main
  """

    @staticmethod
    def Args(parser):
        CloneGA.Args(parser)
        parser.add_argument('--use-full-gcloud-path', action='store_true', help='If provided, use the full gcloud path for the git credential.helper. Using the full path means that gcloud does not need to be in the path for future git operations on the repository.')

    def UseFullGcloudPath(self, args):
        """Use value of --use-full-gcloud-path argument in beta and alpha."""
        return args.use_full_gcloud_path

    def ActionIfMirror(self, project, repo, mirror_url):
        """Raises an exception if the repository is a mirror."""
        message = 'Repository "{repo}" in project "{prj}" is a mirror. Clone the mirrored repository directly with \n$ git clone {url}'.format(repo=repo, prj=project, url=mirror_url)
        raise c_exc.InvalidArgumentException('REPOSITORY_NAME', message)