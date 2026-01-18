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
def _Instances(appinclude):
    """Determines the number of `manual_scaling.instances` sets.

      Args:
        appinclude: The include for which you want to determine the number of
            `manual_scaling.instances` sets.

      Returns:
        The number of instances as an integer. If the value of
        `manual_scaling.instances` evaluates to False (e.g. 0 or None), then
        return 0.
      """
    if appinclude.manual_scaling:
        if appinclude.manual_scaling.instances:
            return int(appinclude.manual_scaling.instances)
    return 0