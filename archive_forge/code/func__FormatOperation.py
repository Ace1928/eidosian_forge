from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _FormatOperation(self, op_id):
    res = resources.REGISTRY.Parse(op_id, params={'appsId': self.project}, collection='appengine.apps.operations')
    return res.RelativeName()