from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import requests
from six.moves import urllib
def _BuildErrorIdentifier(resource_identifier):
    """Builds error identifier from inputed data."""
    return collections.OrderedDict([(key.singular, value) for key, value in resource_identifier.items()])