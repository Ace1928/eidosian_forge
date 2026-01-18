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
def IsAscending(self):
    return self.direction != 'desc'