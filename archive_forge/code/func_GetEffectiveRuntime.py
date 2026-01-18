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
def GetEffectiveRuntime(self):
    """Returns the app's runtime, resolving VMs to the underlying `vm_runtime`.

    Returns:
      The effective runtime: The value of `beta/vm_settings.vm_runtime` if
      `runtime` is `vm`, or `runtime` otherwise.
    """
    if self.runtime == 'vm' and hasattr(self, 'vm_settings') and (self.vm_settings is not None):
        return self.vm_settings.get('vm_runtime')
    if self.runtime == 'vm' and hasattr(self, 'beta_settings') and (self.beta_settings is not None):
        return self.beta_settings.get('vm_runtime')
    return self.runtime