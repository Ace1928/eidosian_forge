from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def ShouldUseYaml(error):
    if hasattr(error, 'code') and (error.code == 'ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS' or error.code == 'ZONE_RESOURCE_POOL_EXHAUSTED' or error.code == 'QUOTA_EXCEEDED'):
        return True
    return False