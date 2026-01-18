from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def ParseParametersFile(path):
    """Reads a JSON file specified by path and returns its contents as a string."""
    with files.FileReader(path) as parameters_file:
        parameters = json.load(parameters_file)
        results = [collections.OrderedDict(sorted(param.items())) for param in parameters]
        return json.dumps(results)