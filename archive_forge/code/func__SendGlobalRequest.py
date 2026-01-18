from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def _SendGlobalRequest(client, project, url_map, load_balancing_scheme):
    return client.apitools_client.urlMaps.Validate(_MakeGlobalRequest(client, project, url_map, load_balancing_scheme))