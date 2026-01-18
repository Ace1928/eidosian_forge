from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def NormalizeVmSettings(self):
    """Normalizes VM settings."""
    if self.IsVm():
        if not self.vm_settings:
            self.vm_settings = VmSettings()
        if 'vm_runtime' not in self.vm_settings:
            self.SetEffectiveRuntime(self.runtime)
        if hasattr(self, 'beta_settings') and self.beta_settings:
            for field in ['vm_runtime', 'has_docker_image', 'image', 'module_yaml_path']:
                if field not in self.beta_settings and field in self.vm_settings:
                    self.beta_settings[field] = self.vm_settings[field]