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
def AddJobResourceArg(parser, verb, api_version):
    concept_parsers.ConceptParser.ForResource('job', _GetJobResourceSpec(api_version), 'The ID of the job to {0}.'.format(verb), required=True).AddToParser(parser)