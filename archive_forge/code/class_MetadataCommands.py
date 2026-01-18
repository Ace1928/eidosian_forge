import os
import sys
from troveclient.compat import common
class MetadataCommands(common.AuthedCommandsBase):
    """Commands to create/update/replace/delete/show metadata for an instance
    """
    params = ['instance_id', 'metadata']

    def show(self):
        """Show instance metadata."""
        self._require('instance_id')
        self._pretty_print(self.dbaas.metadata.show(self.instance_id))