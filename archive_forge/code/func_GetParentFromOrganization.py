from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.policy_intelligence import orgpolicy_simulator
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetParentFromOrganization(org_id):
    """Returns the parent for orgpolicy simulator based on the organization id."""
    return org_id + '/locations/global'