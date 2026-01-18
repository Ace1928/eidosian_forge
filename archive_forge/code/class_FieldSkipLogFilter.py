import logging
class FieldSkipLogFilter(logging.Filter):
    """Filter field skip log messges.

    At most, one message per field skipped per loop will be passed.
    """

    def __init__(self, name=''):
        super().__init__(name)
        self.seen_msgs = set()

    def filter(self, record):
        """Pass record if not seen."""
        msg = record.getMessage()
        if msg.startswith('Skipping field'):
            retval = msg not in self.seen_msgs
            self.seen_msgs.add(msg)
            return retval
        else:
            return 1