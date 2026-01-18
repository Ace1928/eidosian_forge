from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetWorkerMachineTypeArg(required=False):
    return base.Argument('--worker-machine-type', required=required, default=None, help='Default type of machine to use for workers. If not specified here, defaults to server-specified value.')