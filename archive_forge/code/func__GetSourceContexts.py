import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetSourceContexts(source_dir):
    """Gets the source contexts associated with a directory.

  This function is mostly a wrapper around CalculateExtendedSourceContexts
  which logs a message if the context could not be determined.
  Args:
    source_dir: (String) The directory to inspect.
  Returns:
    [ExtendedSourceContext-compatible json dict] A list of 0 or more source
    contexts.
  """
    try:
        source_contexts = CalculateExtendedSourceContexts(source_dir)
    except GenerateSourceContextError:
        source_contexts = []
    if not source_contexts:
        logging.info('Could not find any remote repositories associated with [%s]. Cloud diagnostic tools may not be able to display the correct source code for this deployment.', source_dir)
    return source_contexts