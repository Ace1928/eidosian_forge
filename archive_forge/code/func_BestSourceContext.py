import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def BestSourceContext(source_contexts):
    """Returns the "best" source context from a list of contexts.

  "Best" is a heuristic that attempts to define the most useful context in
  a Google Cloud Platform application. The most useful context is defined as:

  1. The capture context, if there is one. (I.e., a context with category
     'capture')
  2. The Cloud Repo context, if there is one.
  3. A repo context from another known provider (i.e. github or bitbucket), if
     there is no Cloud Repo context.
  4. The generic git repo context, if not of the above apply.

  If there are two Cloud Repo contexts and one of them is a "capture" context,
  that context is considered best.

  If two Git contexts come from the same provider, they will be evaluated based
  on remote name: "origin" is the best name, followed by the name that comes
  last alphabetically.

  If all of the above does not resolve a tie, the tied context that is
  earliest in the source_contexts list wins.

  Args:
    source_contexts: A list of extended source contexts.
  Returns:
    A single source context, or None if source_contexts is empty.
  Raises:
    KeyError if any extended source context is malformed.
  """
    source_context = None
    best_type = None
    best_remote_name = None
    for ext_ctx in source_contexts:
        candidate = ext_ctx['context']
        labels = ext_ctx.get('labels', {})
        context_type = _GetContextType(candidate, labels)
        if best_type and context_type < best_type:
            continue
        remote_name = labels.get('remote_name')
        if context_type == best_type and (not _IsRemoteBetter(remote_name, best_remote_name)):
            continue
        source_context = candidate
        best_remote_name = remote_name
        best_type = context_type
    return source_context