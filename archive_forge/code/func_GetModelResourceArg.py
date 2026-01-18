from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import functools
import itertools
import sys
import textwrap
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.ml_engine import constants
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetModelResourceArg(positional=True, required=False, verb=''):
    """Add a resource argument for AI Platform model.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    positional: bool, if True, means that the model is a positional rather
    required: bool, if True means that argument is required.
    verb: str, the verb to describe the resource, such as 'to update'.

  Returns:
    An argparse.ArgumentParse object.
  """
    if positional:
        name = 'model'
    else:
        name = '--model'
    return concept_parsers.ConceptParser.ForResource(name, GetModelResourceSpec(), 'The AI Platform model {}.'.format(verb), required=required)