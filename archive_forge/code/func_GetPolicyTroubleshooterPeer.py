from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def GetPolicyTroubleshooterPeer(self, destination_ip=None, destination_port=None):
    return self.messages.GoogleCloudPolicytroubleshooterIamV3ConditionContextPeer(ip=destination_ip, port=destination_port)