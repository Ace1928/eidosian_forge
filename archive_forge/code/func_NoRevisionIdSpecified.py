from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub.util import InvalidArgumentError
def NoRevisionIdSpecified():
    return InvalidArgumentError('The schema name must include a revision-id of the format: SCHEMA_NAME@REVISION_ID.')