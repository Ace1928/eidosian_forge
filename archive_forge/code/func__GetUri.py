from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.calliope import base
def _GetUri(resource):
    try:
        return APPENGINE_PATH_START + resource.instance.name
    except AttributeError:
        return APPENGINE_PATH_START + resource['instance']['name']