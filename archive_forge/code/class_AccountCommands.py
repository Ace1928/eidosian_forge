import json
import os
import sys
from troveclient.compat import common
class AccountCommands(common.AuthedCommandsBase):
    """Commands to list account info."""
    params = ['id']

    def list(self):
        """List all accounts with non-deleted instances."""
        self._pretty_print(self.dbaas.accounts.index)

    def get(self):
        """List details for the account provided."""
        self._require('id')
        self._pretty_print(self.dbaas.accounts.show, self.id)