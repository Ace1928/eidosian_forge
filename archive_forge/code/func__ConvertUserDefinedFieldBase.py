from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def _ConvertUserDefinedFieldBase(base):
    return {'ipv4': 'IPV4', 'ipv6': 'IPV6', 'tcp': 'TCP', 'udp': 'UDP'}.get(base, base)