from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def BuildDatabaseFilter(instance, database):
    database_ref = resources.REGISTRY.Parse(database, params={'projectsId': properties.VALUES.core.project.GetOrFail, 'instancesId': instance}, collection='spanner.projects.instances.databases')
    return 'metadata.database:"{}"'.format(database_ref.RelativeName())