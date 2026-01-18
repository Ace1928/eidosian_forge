from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
class ResourceMapUpdateUnmatchedError(ResourceMapUpdateError):
    """Exception when an update map has unmatched members."""

    def __init__(self, unmatched):
        super(ResourceMapUpdateUnmatchedError, self).__init__('Registered update map has unmatched members. Please fix error leading to mismatch or add to allowlist: \n {}'.format(unmatched))