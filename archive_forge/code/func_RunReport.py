from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
def RunReport(self):
    result = self.client.services.GenerateConfigReport(self.ConstructRequestMessage())
    if not result:
        return None
    if not result.changeReports:
        return []
    return result.changeReports[0]