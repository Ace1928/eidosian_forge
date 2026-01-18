from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddSubstitutionUpdatingFlags(argument_group):
    """Adds substitution updating flags to the given argument group.

  Args:
    argument_group: argparse argument group to which the substitution updating
      flags flag will be added.
  """
    argument_group.add_argument('--update-substitutions', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='Update or add to existing substitutions.\nSubstitutions are parameters to be substituted or add in the build specification.\n\nFor example (using some nonsensical substitution keys; all keys must begin with\nan underscore):\n\n  $ gcloud builds triggers update ...\n      --update-substitutions _FAVORITE_COLOR=blue,_NUM_CANDIES=10\n\nThis will add the provided substitutions to the existing substitutions and\nresults in a build where every occurrence of ```${_FAVORITE_COLOR}```\nin certain fields is replaced by "blue", and similarly for ```${_NUM_CANDIES}```\nand "10".\n\nOnly the following built-in variables can be specified with the\n`--substitutions` flag: REPO_NAME, BRANCH_NAME, TAG_NAME, REVISION_ID,\nCOMMIT_SHA, SHORT_SHA.\n\nFor more details, see:\nhttps://cloud.google.com/build/docs/build-config-file-schema#substitutions\n')
    argument_group.add_argument('--clear-substitutions', action='store_true', help='Clear existing substitutions.')
    argument_group.add_argument('--remove-substitutions', metavar='KEY', type=arg_parsers.ArgList(), help='Remove existing substitutions if present.')