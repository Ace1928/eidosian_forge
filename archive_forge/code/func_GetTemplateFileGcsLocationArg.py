from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetTemplateFileGcsLocationArg(required=False):
    return base.Argument('--template-file-gcs-location', required=required, default=None, type=arg_parsers.RegexpValidator('^gs://.*', "Must begin with 'gs://'"), help='Location of the template file or container spec file in Google Cloud Storage.')