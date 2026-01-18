from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
def IsReadyForReport(self):
    return self.config is not None or self.swagger_path is not None or self.config_id is not None or self.config_use_active_id