from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetAdditionalPackageTypesFlag():
    return base.Argument('--additional-package-types', type=arg_parsers.ArgList(choices=_PACKAGE_TYPE_CHOICES, element_type=lambda package_type: package_type.upper()), metavar='ADDITIONAL_PACKAGE_TYPES', help='A comma-separated list of package types to scan in addition to OS packages.')