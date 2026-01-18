import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetContextType(context, labels):
    """Returns the _ContextType for the input extended source context.

  Args:
    context: A source context dict.
    labels: A dict containing the labels associated with the context.
  Returns:
    The context type.
  """
    if labels.get('category') == CAPTURE_CATEGORY:
        return _ContextType.SOURCE_CAPTURE
    git_context = context.get('git')
    if git_context:
        return _GetGitContextTypeFromDomain(git_context.get('url'))
    if 'cloudRepo' in context:
        return _ContextType.CLOUD_REPO
    return _ContextType.OTHER