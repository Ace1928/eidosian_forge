import paste.util.threadinglocal as threadinglocal
def _current_obj_restoration(self):
    request_id = restorer.in_restoration()
    if request_id:
        return restorer.get_saved_proxied_obj(self, request_id)
    return self._current_obj_orig()