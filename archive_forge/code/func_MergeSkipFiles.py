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
@staticmethod
def MergeSkipFiles(skip_files_one, skip_files_two):
    """Merges two `skip_files` directives.

    Args:
      skip_files_one: The first `skip_files` element that you want to merge.
      skip_files_two: The second `skip_files` element that you want to merge.

    Returns:
      A list of regular expressions that are merged.
    """
    if skip_files_one == SKIP_NO_FILES:
        return skip_files_two
    if skip_files_two == SKIP_NO_FILES:
        return skip_files_one
    return validation.RegexStr().Validate([skip_files_one, skip_files_two], SKIP_FILES)