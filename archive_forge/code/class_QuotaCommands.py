import json
import os
import sys
from troveclient.compat import common
class QuotaCommands(common.AuthedCommandsBase):
    """List and update quota limits for a tenant."""
    params = ['id', 'instances', 'volumes', 'backups']

    def list(self):
        """List all quotas for a tenant."""
        self._require('id')
        self._pretty_print(self.dbaas.quota.show, self.id)

    def update(self):
        """Update quota limits for a tenant."""
        self._require('id')
        self._pretty_print(self.dbaas.quota.update, self.id, dict(((param, getattr(self, param)) for param in self.params if param != 'id')))