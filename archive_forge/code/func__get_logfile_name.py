from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from surface.spanner.samples import init as samples_init
def _get_logfile_name(appname):
    return '{}-backend.log'.format(appname)