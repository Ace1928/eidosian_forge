from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
def ConstructRequestMessage(self):
    old_config_value = self.old_config.ConstructConfigValue(self.messages.GenerateConfigReportRequest.OldConfigValue)
    new_config_value = self.new_config.ConstructConfigValue(self.messages.GenerateConfigReportRequest.NewConfigValue)
    return self.messages.GenerateConfigReportRequest(oldConfig=old_config_value, newConfig=new_config_value)