from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def GetSubcollection(self, collection_name):
    name = self.full_name
    if collection_name.startswith(name):
        return collection_name[len(name) + 1:]
    raise KeyError('{0} does not exist in {1}'.format(collection_name, name))