from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.core import exceptions as gcloud_exceptions
import six
def _rebuild_spec(self, spec: messages.Message) -> messages.Message:
    """Rebuilds the spec to only include information from policycontroller.

    This is necessary so that feature-level values managed by Hub are not
    unintentionally overwritten (i.e. 'origin').

    Args:
      spec: The spec found by querying the API.

    Returns:
      MembershipFeatureSpec with only policycontroller values, leaving off
      other top-level data.
    """
    return self.messages.MembershipFeatureSpec(policycontroller=spec.policycontroller)