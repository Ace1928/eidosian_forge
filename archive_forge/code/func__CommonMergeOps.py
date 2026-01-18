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
@classmethod
def _CommonMergeOps(cls, one, two):
    """This function performs common merge operations.

    Args:
      one: The first object that you want to merge.
      two: The second object that you want to merge.

    Returns:
      An updated `one` object containing all merged data.
    """
    AppInclude.MergeManualScaling(one, two)
    one.admin_console = AdminConsole.Merge(one.admin_console, two.admin_console)
    one.vm = two.vm or one.vm
    one.vm_settings = VmSettings.Merge(one.vm_settings, two.vm_settings)
    if hasattr(one, 'beta_settings'):
        one.beta_settings = BetaSettings.Merge(one.beta_settings, two.beta_settings)
    one.env_variables = EnvironmentVariables.Merge(one.env_variables, two.env_variables)
    one.skip_files = cls.MergeSkipFiles(one.skip_files, two.skip_files)
    return one