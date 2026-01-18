from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
def SetSwagger(self, path, contents):
    self.config = None
    self.swagger_path = path
    self.swagger_contents = contents
    self.config_id = None
    self.config_use_active_id = False