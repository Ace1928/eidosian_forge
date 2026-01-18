from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources as fleet_resources
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
def _AddSecurityPostureMode(self, security_posture_config_group: parser_arguments.ArgumentInterceptor):
    security_posture_config_group.add_argument('--security-posture', choices=['disabled', 'standard'], default=None, help=textwrap.dedent('          To apply standard security posture to clusters in the fleet,\n\n            $ {command} --security-posture=standard\n\n          '))