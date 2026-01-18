from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def AddGuestOsFeaturesArgForImport(parser, messages):
    """Add the guest-os-features arg for import commands."""
    AddGuestOsFeaturesArg(parser, messages, supported_features=[messages.GuestOsFeature.TypeValueValuesEnum.UEFI_COMPATIBLE.name])