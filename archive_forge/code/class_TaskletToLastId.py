from __future__ import nested_scopes
import weakref
import sys
from _pydevd_bundle.pydevd_comm import get_global_debugger
from _pydevd_bundle.pydevd_constants import call_only_once
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import update_custom_frame, remove_custom_frame, add_custom_frame
import stackless  # @UnresolvedImport
from _pydev_bundle import pydev_log
class TaskletToLastId:
    """
    So, why not a WeakKeyDictionary?
    The problem is that removals from the WeakKeyDictionary will create a new tasklet (as it adds a callback to
    remove the key when it's garbage-collected), so, we can get into a recursion.
    """

    def __init__(self):
        self.tasklet_ref_to_last_id = {}
        self._i = 0

    def get(self, tasklet):
        return self.tasklet_ref_to_last_id.get(weakref.ref(tasklet))

    def __setitem__(self, tasklet, last_id):
        self.tasklet_ref_to_last_id[weakref.ref(tasklet)] = last_id
        self._i += 1
        if self._i % 100 == 0:
            for tasklet_ref in list(self.tasklet_ref_to_last_id.keys()):
                if tasklet_ref() is None:
                    del self.tasklet_ref_to_last_id[tasklet_ref]