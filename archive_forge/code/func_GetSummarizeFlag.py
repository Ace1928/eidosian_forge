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
def GetSummarizeFlag():
    return base.Argument('--summarize', action='store_true', required=False, help='      Summarize job output in a set of human readable tables instead of\n      rendering the entire resource as json or yaml. The tables currently rendered\n      are:\n\n      * `Job Overview`: Overview of job including, jobId, status and create time.\n      * `Training Input Summary`: Summary of input for a training job including\n         region, main training python module and scale tier.\n      * `Training Output Summary`: Summary of output for a training job including\n         the amount of ML units consumed by the job.\n      * `Training Output Trials`: Summary of hyperparameter trials run for a\n         hyperparameter tuning training job.\n      * `Predict Input Summary`: Summary of input for a prediction job including\n         region, model verion and output path.\n      * `Predict Output Summary`: Summary of output for a prediction job including\n         prediction count and output path.\n\n      This flag overrides the `--format` flag. If\n      both are present on the command line, a warning is displayed.\n      ')