import logging
def _flush_if_needed(self):
    if len(self._items_buffer) >= self._flush_amount:
        self._flush()