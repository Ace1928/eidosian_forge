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
def MergeAppIncludes(cls, appinclude_one, appinclude_two):
    """Merges the non-referential state of the provided `AppInclude`.

    That is, `builtins` and `includes` directives are not preserved, but any
    static objects are copied into an aggregate `AppInclude` object that
    preserves the directives of both provided `AppInclude` objects.

    `appinclude_one` is updated to be the merged result in this process.

    Args:
      appinclude_one: First `AppInclude` to merge.
      appinclude_two: Second `AppInclude` to merge.

    Returns:
      `AppInclude` object that is the result of merging the static directives of
      `appinclude_one` and `appinclude_two`. An updated version of
      `appinclude_one` is returned.
    """
    if not appinclude_one or not appinclude_two:
        return appinclude_one or appinclude_two
    if appinclude_one.handlers:
        if appinclude_two.handlers:
            appinclude_one.handlers.extend(appinclude_two.handlers)
    else:
        appinclude_one.handlers = appinclude_two.handlers
    return cls._CommonMergeOps(appinclude_one, appinclude_two)