from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetServiceRelativeName(self, service_name):
    res = resources.REGISTRY.Parse(service_name, params={'appsId': self.project}, collection='appengine.apps.services')
    return res.RelativeName()