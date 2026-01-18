from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddHttpRelatedResponseArg(parser):
    """Adds parser argument for HTTP response field."""
    parser.add_argument('--response', help='      When empty, status code of the response determines health. When not empty,\n      presence of specified string in first 1024 characters of response body\n      determines health. Only ASCII characters allowed.\n      ')