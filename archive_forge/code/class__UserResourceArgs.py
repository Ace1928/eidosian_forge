from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
class _UserResourceArgs(object):
    """Contains user flag values affecting cloud settings."""

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