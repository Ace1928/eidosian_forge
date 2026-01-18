import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _to_project(self, project):
    """
        Return a Project object from the JSON-response dictionary.

        :param  project: The dictionary describing the project.
        :type   project: ``dict``

        :return: Project object
        :rtype: :class:`GCEProject`
        """
    extra = {}
    extra['selfLink'] = project.get('selfLink')
    extra['creationTimestamp'] = project.get('creationTimestamp')
    extra['description'] = project.get('description')
    metadata = project['commonInstanceMetadata'].get('items')
    if 'commonInstanceMetadata' in project:
        extra['commonInstanceMetadata'] = project['commonInstanceMetadata']
    if 'usageExportLocation' in project:
        extra['usageExportLocation'] = project['usageExportLocation']
    return GCEProject(id=project['id'], name=project['name'], metadata=metadata, quotas=project.get('quotas'), driver=self, extra=extra)