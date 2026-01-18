import paste.util.threadinglocal as threadinglocal
def _push_object_restoration(self, obj):
    if not restorer.in_restoration():
        self._push_object_orig(obj)