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
class BetaSettings(VmSettings):
    """Class for Beta (internal or unreleased) settings.

  This class is meant to replace `VmSettings` eventually.

  Note:
      All new beta settings must be registered in `shared_constants.py`.

  These settings are not validated further here. The settings are validated on
  the server side.
  """

    @classmethod
    def Merge(cls, beta_settings_one, beta_settings_two):
        """Merges two `BetaSettings` instances.

    Args:
      beta_settings_one: The first `BetaSettings` instance, or `None`.
      beta_settings_two: The second `BetaSettings` instance, or `None`.

    Returns:
      The merged `BetaSettings` instance, or `None` if both input instances are
      `None` or empty.
    """
        merged = VmSettings.Merge(beta_settings_one, beta_settings_two)
        return BetaSettings(**merged.ToDict()) if merged else None