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
def AddGcsSourceStagingDirFlag(parser, hidden=False):
    """Add a GCS directory flag for staging the build."""
    parser.add_argument('--gcs-source-staging-dir', hidden=hidden, help="A directory in Google Cloud Storage to copy the source used for staging the build. If the specified bucket does not exist, Cloud Build will create one. If you don't set this field, ```gs://[PROJECT_ID]_cloudbuild/source``` is used or ```gs://[PROJECT_ID]_[builds/region]_cloudbuild/source``` is used when you set `--default-buckets-behavior` to `REGIONAL_USER_OWNED_BUCKET` and `builds/region` is not `global`.")