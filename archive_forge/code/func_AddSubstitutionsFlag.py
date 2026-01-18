from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddSubstitutionsFlag(parser, hidden=False):
    """Add a substitutions flag."""
    parser.add_argument('--substitutions', hidden=hidden, metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='Parameters to be substituted in the build specification.\n\nFor example (using some nonsensical substitution keys; all keys must begin with\nan underscore):\n\n    $ gcloud builds submit . --config config.yaml \\\n        --substitutions _FAVORITE_COLOR=blue,_NUM_CANDIES=10\n\nThis will result in a build where every occurrence of ```${_FAVORITE_COLOR}```\nin certain fields is replaced by "blue", and similarly for ```${_NUM_CANDIES}```\nand "10".\n\nOnly the following built-in variables can be specified with the\n`--substitutions` flag: REPO_NAME, BRANCH_NAME, TAG_NAME, REVISION_ID,\nCOMMIT_SHA, SHORT_SHA.\n\nFor more details, see:\nhttps://cloud.google.com/cloud-build/docs/api/build-requests#substitutions\n')