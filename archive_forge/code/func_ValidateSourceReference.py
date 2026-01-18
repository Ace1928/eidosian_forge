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
def ValidateSourceReference(ref):
    """Determines if a source reference is valid.

  Args:
    ref: A source reference in the following format:
        `[repository_uri#]revision`.

  Raises:
    ValidationError: If the reference is malformed.
  """
    repo_revision = ref.split('#', 1)
    revision_id = repo_revision[-1]
    if not re.match(SOURCE_REVISION_RE_STRING, revision_id):
        raise validation.ValidationError('Bad revision identifier: %s' % revision_id)
    if len(repo_revision) == 2:
        uri = repo_revision[0]
        if not re.match(SOURCE_REPO_RE_STRING, uri):
            raise validation.ValidationError('Bad repository URI: %s' % uri)