import logging
def _add_request_and_process(self, request):
    if self._overwrite_by_pkeys:
        self._remove_dup_pkeys_request_if_any(request)
    self._items_buffer.append(request)
    self._flush_if_needed()