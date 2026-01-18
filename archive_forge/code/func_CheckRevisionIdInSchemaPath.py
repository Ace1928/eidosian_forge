from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub.util import InvalidArgumentError
def CheckRevisionIdInSchemaPath(schema_ref):
    find_id = schema_ref.split('@')
    return len(find_id) > 1