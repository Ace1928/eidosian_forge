import json
import os
import sys
from troveclient.compat import common
class StorageCommands(common.AuthedCommandsBase):
    """Commands to list devices info."""
    params = []

    def list(self):
        """List details for the storage device."""
        self._pretty_list(self.dbaas.storage.index)