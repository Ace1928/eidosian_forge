from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _RemoveEncoding(self, elements):
    """Removes encoding for each element in the list.

    This causes inconsistencies in the scenario test when the output
    looks like [u'domain'] instead of ['domain']

    Args:
      elements: list

    Returns:
      list[str], encoding removed from each element.
    """
    return [str(x) for x in elements]