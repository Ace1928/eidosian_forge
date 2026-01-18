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
def ErrorOnPositionForAppInfo(self):
    """Raises an error if position is specified outside of AppInclude objects.

    Raises:
      PositionUsedInAppYamlHandler: If the `position` attribute is specified for
          an `app.yaml` file instead of an `include.yaml` file.
    """
    if self.position:
        raise appinfo_errors.PositionUsedInAppYamlHandler('The position attribute was specified for this handler, but this is an app.yaml file.  Position attribute is only valid for include.yaml files.')