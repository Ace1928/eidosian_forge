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
class VmSettings(validation.ValidatedDict):
    """Class for VM settings.

  The settings are not further validated here. The settings are validated on
  the server side.
  """
    KEY_VALIDATOR = validation.Regex('[a-zA-Z_][a-zA-Z0-9_]*')
    VALUE_VALIDATOR = str

    @classmethod
    def Merge(cls, vm_settings_one, vm_settings_two):
        """Merges two `VmSettings` instances.

    If a variable is specified by both instances, the value from
    `vm_settings_one` is used.

    Args:
      vm_settings_one: The first `VmSettings` instance, or `None`.
      vm_settings_two: The second `VmSettings` instance, or `None`.

    Returns:
      The merged `VmSettings` instance, or `None` if both input instances are
      `None` or empty.
    """
        result_vm_settings = (vm_settings_two or {}).copy()
        result_vm_settings.update(vm_settings_one or {})
        return VmSettings(**result_vm_settings) if result_vm_settings else None