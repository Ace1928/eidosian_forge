from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def DescribeEntities(self, name, args=None):
    """Describes database entities in a conversion worksapce.

    Args:
      name: str, the name for conversion worksapce being described.
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      Described entities for the conversion worksapce.
    """
    entity_result = []
    page_size = 4000
    page_token = str()
    describe_req = self._GetDescribeEntitiesRequest(name, page_size, page_token, args)
    while True:
        response = self._service.DescribeDatabaseEntities(describe_req)
        entities = response.databaseEntities
        for entity in entities:
            entity_result.append({'parentEntity': entity.parentEntity, 'shortName': entity.shortName, 'tree': entity.tree, 'entityType': six.text_type(entity.entityType).replace('DATABASE_ENTITY_TYPE_', '')})
        if not response.nextPageToken:
            break
        describe_req.pageToken = response.nextPageToken
    return entity_result