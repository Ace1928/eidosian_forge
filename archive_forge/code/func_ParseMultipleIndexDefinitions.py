from __future__ import absolute_import
from ruamel import yaml
import copy
import itertools
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_object
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
def ParseMultipleIndexDefinitions(document):
    """Parse multiple index definitions documents from a string or stream.

  Args:
    document: Yaml document as a string or file-like stream.

  Returns:
    A list of datastore_index.IndexDefinitions objects, one for each document.
  """
    return yaml_object.BuildObjects(IndexDefinitions, document)