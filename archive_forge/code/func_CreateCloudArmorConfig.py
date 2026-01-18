from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateCloudArmorConfig(client, args):
    """Returns a SecurityPolicyCloudArmorConfig message."""
    messages = client.messages
    cloud_armor_config = None
    if args.enable_ml is not None:
        cloud_armor_config = messages.SecurityPolicyCloudArmorConfig(enableMl=args.enable_ml)
    return cloud_armor_config