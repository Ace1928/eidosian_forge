from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
def GetTypeUrl(self):
    if self.config:
        return ConfigReporterValue.SERVICE_CONFIG_TYPE_URL
    elif self.swagger_path and self.swagger_contents:
        return ConfigReporterValue.CONFIG_SOURCE_TYPE_URL
    elif self.config_id or self.config_use_active_id:
        return ConfigReporterValue.CONFIG_REF_TYPE_URL