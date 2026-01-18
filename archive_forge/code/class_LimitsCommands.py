import os
import sys
from troveclient.compat import common
class LimitsCommands(common.AuthedCommandsBase):
    """Show the rate limits and absolute limits."""

    def list(self):
        """List the rate limits and absolute limits."""
        self._pretty_list(self.dbaas.limits.list)