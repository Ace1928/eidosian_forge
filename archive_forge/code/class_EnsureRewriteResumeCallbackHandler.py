from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class EnsureRewriteResumeCallbackHandler(object):
    """Test callback handler for ensuring a rewrite operation resumed."""

    def __init__(self, required_byte):
        self._required_byte = required_byte

    def call(self, total_bytes_rewritten, unused_total_size):
        """Exits if the total bytes rewritten is less than expected."""
        if total_bytes_rewritten <= self._required_byte:
            raise RewriteHaltException('Rewrite did not resume; %s bytes written, but %s bytes should have already been written.' % (total_bytes_rewritten, self._required_byte))