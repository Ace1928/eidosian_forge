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
def DetailsForLatest(self, component_ids):
    """Gets the schema.Component objects for all ids from the latest snapshot.

    Args:
      component_ids: list of str, The component ids to get.

    Returns:
      A list of schema.Component objects sorted by component display name.
    """
    return sorted(self.latest.ComponentsFromIds(component_ids), key=lambda c: c.details.display_name)