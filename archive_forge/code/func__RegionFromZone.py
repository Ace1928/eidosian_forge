from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import exceptions as base_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _RegionFromZone(zone):
    return '-'.join(zone.split('-')[:2])