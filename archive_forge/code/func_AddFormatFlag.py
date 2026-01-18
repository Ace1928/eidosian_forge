from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddFormatFlag(parser):
    parser.add_argument('--resource-format', choices=['krm', 'terraform'], help='Format of the configuration to export. Available configuration formats are Kubernetes Resource Model YAML (krm) or Terraform HCL (terraform). Command defaults to "krm".')