from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def _ConvertExceedAction(action):
    return {'deny-403': 'deny(403)', 'deny-404': 'deny(404)', 'deny-429': 'deny(429)', 'deny-502': 'deny(502)'}.get(action, action)