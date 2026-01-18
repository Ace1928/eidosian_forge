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
def MergeManualScaling(cls, appinclude_one, appinclude_two):
    """Takes the greater of `<manual_scaling.instances>` from the arguments.

    `appinclude_one` is mutated to be the merged result in this process.

    Also, this function must be updated if `ManualScaling` gets additional
    fields.

    Args:
      appinclude_one: The first object to merge. The object must have a
          `manual_scaling` field that contains a `ManualScaling()`.
      appinclude_two: The second object to merge. The object must have a
          `manual_scaling` field that contains a `ManualScaling()`.
    Returns:
      An object that is the result of merging
      `appinclude_one.manual_scaling.instances` and
      `appinclude_two.manual_scaling.instances`; this is returned as a revised
      `appinclude_one` object after the mutations are complete.
    """

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
    if _Instances(appinclude_one) or _Instances(appinclude_two):
        instances = max(_Instances(appinclude_one), _Instances(appinclude_two))
        appinclude_one.manual_scaling = ManualScaling(instances=str(instances))
    return appinclude_one