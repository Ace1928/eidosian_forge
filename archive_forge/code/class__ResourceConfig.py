from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class _ResourceConfig(object):
    """Holder for generic resource fields.

  Attributes:
    acl_file_path (None|str): Path to file with ACL settings.
    acl_grants_to_add (None|list[dict]): Contains API representations of ACL.
      For GCS, this looks like `{ 'entity': ENTITY, 'role': GRANT }`.
    acl_grants_to_remove: (None|list[str]): Identifier of entity to remove
      access for. Can be user, group, project, or keyword like "All".
  """

    def __init__(self, acl_file_path=None, acl_grants_to_add=None, acl_grants_to_remove=None):
        """Initializes class, binding flag values to it."""
        self.acl_file_path = acl_file_path
        self.acl_grants_to_add = acl_grants_to_add
        self.acl_grants_to_remove = acl_grants_to_remove

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.acl_file_path == other.acl_file_path and self.acl_grants_to_add == other.acl_grants_to_add and (self.acl_grants_to_remove == other.acl_grants_to_remove)

    def __repr__(self):
        return debug_output.generic_repr(self)