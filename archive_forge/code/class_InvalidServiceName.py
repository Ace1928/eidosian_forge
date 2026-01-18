from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
class InvalidServiceName(exceptions.Error):
    """Error when a given serviceName is invalid."""