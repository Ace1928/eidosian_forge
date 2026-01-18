from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidFormatError(ParseError):

    def __init__(self, path, reason, message_class):
        valid_fields = [f.name for f in message_class.all_fields()]
        super(InvalidFormatError, self).__init__(path, 'Invalid format: {}\n\nAn authorized orgs desc file is a YAML-formatted list of authorized orgs descs, which are YAML objects with the fields [{}]. For example:\n\n- name: my_authorized_orgs\n  authorizationType: AUTHORIZATION_TYPE_TRUST.\n  assetType: ASSET_TYPE_DEVICE.\n  authorizationDirection: AUTHORIZATION_DIRECTION_TO.\n  orgs:\n  - organizations/123456789\n  - organizations/234567890\n'.format(reason, ', '.join(valid_fields)))