from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class EnsureRewriteRestartCallbackHandler(object):
    """Test callback handler for ensuring a rewrite operation restarted."""

    def __init__(self, required_byte):
        self._required_byte = required_byte
        self._got_restart_bytes = False

    def call(self, total_bytes_rewritten, unused_total_size):
        """Exits if the total bytes rewritten is greater than expected."""
        if not self._got_restart_bytes:
            if total_bytes_rewritten <= self._required_byte:
                self._got_restart_bytes = True
            else:
                raise RewriteHaltException('Rewrite did not restart; %s bytes written, but no more than %s bytes should have already been written.' % (total_bytes_rewritten, self._required_byte))