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
def FilterDuplicatesArm(self, component_ids):
    """Filter out x86_64 components that are available in arm versions."""
    return set((i for i in component_ids if not ('darwin-x86_64' in i and i.replace('x86_64', 'arm') in component_ids)))