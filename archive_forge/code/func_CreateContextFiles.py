import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def CreateContextFiles(output_dir, source_contexts, overwrite=False, source_dir=None):
    """Creates source context file in the given directory if possible.

  Currently, only source-context.json file will be produced.

  Args:
    output_dir: (String) The directory to create the files (usually the yaml
        directory).
    source_contexts:  ([ExtendedSourceContext-compatible json dict])
        A list of json-serializable dicts containing source contexts. If None
        or empty, source context will be inferred from source_dir.
    overwrite: (boolean) If true, silently replace any existing file.
    source_dir: (String) The location of the source files, for inferring
        source contexts when source_contexts is empty or None. If not
        specified, output_dir will be used instead.
  Returns:
    ([String]) A list containing the names of the files created. If there are
    no source contexts found, or if the contexts files could not be created, the
    result will be an empty.
  """
    if not source_contexts:
        source_contexts = _GetSourceContexts(source_dir or output_dir)
        if not source_contexts:
            return []
    created = []
    for context_filename, context_object in [(CONTEXT_FILENAME, BestSourceContext(source_contexts))]:
        context_filename = os.path.join(output_dir, context_filename)
        try:
            if overwrite or not os.path.exists(context_filename):
                with open(context_filename, 'w') as f:
                    json.dump(context_object, f)
                created.append(context_filename)
        except IOError as e:
            logging.warn('Could not generate [%s]: %s', context_filename, e)
    return created