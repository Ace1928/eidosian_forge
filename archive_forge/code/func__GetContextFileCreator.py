import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetContextFileCreator(output_dir, contexts):
    """Creates a creator function for an old-style source context file.

  Args:
    output_dir: (String) The name of the directory in which to generate the
        file. The file will be named source-context.json.
    contexts: ([dict]) A list of ExtendedSourceContext-compatible dicts for json
        serialization.
  Returns:
    A creator function that will create the file.
  """
    name = os.path.join(output_dir, CONTEXT_FILENAME)
    return _GetJsonFileCreator(name, BestSourceContext(contexts))