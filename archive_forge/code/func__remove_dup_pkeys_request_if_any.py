import logging
def _remove_dup_pkeys_request_if_any(self, request):
    pkey_values_new = self._extract_pkey_values(request)
    for item in self._items_buffer:
        if self._extract_pkey_values(item) == pkey_values_new:
            self._items_buffer.remove(item)
            logger.debug('With overwrite_by_pkeys enabled, skipping request:%s', item)