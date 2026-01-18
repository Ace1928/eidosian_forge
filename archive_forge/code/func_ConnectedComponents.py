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
def ConnectedComponents(self, component_ids, platform_filter=None):
    """Gets all the components that are connected to any of the given ids.

    Connected means in the connected graph of dependencies.  This is basically
    the union of the dependency and consumer closure for the given ids.

    Args:
      component_ids: list of str, The ids of the components to get the
        connected graph of.
      platform_filter: platforms.Platform, A platform that components must
        match to be pulled into the connected graph.

    Returns:
      set of str, All component ids that are connected to the given ids,
      including the given components.
    """
    adjacencies = {c_id: self.__dependencies[c_id] | self.__consumers[c_id] for c_id in self.components}
    component_filter = lambda c: c.platform.Matches(platform_filter)
    return self._ClosureFor(component_ids, adjacencies, component_filter)