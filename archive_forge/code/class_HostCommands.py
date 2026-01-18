import json
import os
import sys
from troveclient.compat import common
class HostCommands(common.AuthedCommandsBase):
    """Commands to list info on hosts."""
    params = ['name']

    def update_all(self):
        """Update all instances on a host."""
        self._require('name')
        self.dbaas.hosts.update_all(self.name)

    def get(self):
        """List details for the specified host."""
        self._require('name')
        self._pretty_print(self.dbaas.hosts.get, self.name)

    def list(self):
        """List all compute hosts."""
        self._pretty_list(self.dbaas.hosts.index)