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
def AddExplainabilityFlags(parser):
    """Add args that configure explainability."""
    base.ChoiceArgument('--explanation-method', choices=['integrated-gradients', 'sampled-shapley', 'xrai'], required=False, help_str='          Enable explanations and select the explanation method to use.\n\n          The valid options are:\n            integrated-gradients: Use Integrated Gradients.\n            sampled-shapley: Use Sampled Shapley.\n            xrai: Use XRAI.\n      ').AddToParser(parser)
    base.Argument('--num-integral-steps', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), default=50, required=False, help='          Number of integral steps for Integrated Gradients. Only valid when\n          `--explanation-method=integrated-gradients` or\n          `--explanation-method=xrai` is specified.\n      ').AddToParser(parser)
    base.Argument('--num-paths', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), default=50, required=False, help='          Number of paths for Sampled Shapley. Only valid when\n          `--explanation-method=sampled-shapley` is specified.\n      ').AddToParser(parser)