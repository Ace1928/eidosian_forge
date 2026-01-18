from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetPolicysimulatorOrgPolicyViolationsPreview(self, name=None, overlay=None, resource_counts=None, state=None, violations_count=None):
    return self.messages.GoogleCloudPolicysimulatorV1OrgPolicyViolationsPreview(name=name, overlay=overlay, resourceCounts=resource_counts, state=state, violationsCount=violations_count)