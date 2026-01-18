from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
def ComponentsFromIds(self, component_ids):
    """Gets the schemas.Component objects for each of the given ids.

    Args:
      component_ids: iterable of str, The ids of the  components to get

    Returns:
      The corresponding schemas.Component objects.
    """
    return set((self.components.get(component_id) for component_id in component_ids))