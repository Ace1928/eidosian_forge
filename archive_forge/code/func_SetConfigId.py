from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
def SetConfigId(self, config_id):
    self.config = None
    self.swagger_path = None
    self.swagger_contents = None
    self.config_id = config_id
    self.config_use_active_id = False