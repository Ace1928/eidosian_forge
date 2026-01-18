from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddCloudRunFileFlag():
    return base.Argument('--from-run-manifest', help='The path to a Cloud Run manifest, which Cloud Deploy will use to generate a skaffold.yaml file for you (for example, foo/bar/service.yaml). The generated Skaffold file will be available in the Google Cloud Storage source staging directory (see --gcs-source-staging-dir flag) after the release is complete.')