import os
import sys
from troveclient.compat import common
class FlavorsCommands(common.AuthedCommandsBase):
    """Command for listing Flavors."""
    params = []

    def list(self):
        """List the available flavors."""
        self._pretty_list(self.dbaas.flavors.list)