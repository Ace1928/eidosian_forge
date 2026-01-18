import sys
import logging
def add_format_prefix(self, prefix):
    """
        Include `prefix` in all future logging statements.
        """
    self.prefix = prefix
    self.streamHandler.setFormatter(self._build_formatter())