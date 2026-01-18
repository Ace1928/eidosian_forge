from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import enum
import functools
import re
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute.regions import service as regions_service
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.util import text
import six
def AddGuestFlushFlag(parser, resource, custom_help=None):
    help_text = '\n  Create an application-consistent {} by informing the OS\n  to prepare for the snapshot process.\n  '.format(resource)
    parser.add_argument('--guest-flush', action='store_true', default=False, help=custom_help if custom_help else help_text)