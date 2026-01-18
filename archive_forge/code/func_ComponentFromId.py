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
def ComponentFromId(self, component_id):
    """Gets the schemas.Component from this snapshot with the given id.

    Args:
      component_id: str, The id component to get.

    Returns:
      The corresponding schemas.Component object.
    """
    return self.components.get(component_id)