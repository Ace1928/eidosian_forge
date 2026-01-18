from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def _GetOperationResourceSpec(api_version):
    return concepts.ResourceSpec('dataproc.projects.regions.operations', api_version=api_version, resource_name='operation', disable_auto_completers=True, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, regionsId=_RegionAttributeConfig(), operationsId=OperationConfig())