from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddCommentControlArg(argument_group):
    """Adds additional argparse flags to a group for comment control options.

  Args:
    argument_group: Argparse argument group to which comment control flag will
      be added.
  """
    argument_group.add_argument('--comment-control', default='COMMENTS_ENABLED', help="Require a repository collaborator or owner to comment '/gcbrun' on a pull request before running the build.", choices={'COMMENTS_DISABLED': 'Do not require comments on Pull Requests before builds are triggered.', 'COMMENTS_ENABLED': 'Enforce that repository owners or collaborators must comment on Pull Requests before builds are triggered.', 'COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY': "Enforce that repository owners or collaborators must comment on external contributors' Pull Requests before builds are triggered."})