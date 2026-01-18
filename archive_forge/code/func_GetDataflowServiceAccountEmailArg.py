from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetDataflowServiceAccountEmailArg(required=False):
    return base.Argument('--dataflow-service-account-email', required=required, default=None, help='Default service account to run the dataflow workers as.')