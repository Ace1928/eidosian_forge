import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def ExtendContextDict(context, category=REMOTE_REPO_CATEGORY, remote_name=None):
    """Converts a source context dict to an ExtendedSourceContext dict.

  Args:
    context: A SourceContext-compatible dict
    category:  string indicating the category of context (either
        CAPTURE_CATEGORY or REMOTE_REPO_CATEGORY)
    remote_name: The name of the remote in git.
  Returns:
    An ExtendedSourceContext-compatible dict.
  """
    labels = {'category': category}
    if remote_name:
        labels['remote_name'] = remote_name
    return {'context': context, 'labels': labels}