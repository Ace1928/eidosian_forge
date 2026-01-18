from __future__ import nested_scopes
import weakref
import sys
from _pydevd_bundle.pydevd_comm import get_global_debugger
from _pydevd_bundle.pydevd_constants import call_only_once
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import update_custom_frame, remove_custom_frame, add_custom_frame
import stackless  # @UnresolvedImport
from _pydev_bundle import pydev_log
def _schedule_callback(prev, next):
    """
        Called when a context is stopped or a new context is made runnable.
        """
    try:
        if not prev and (not next):
            return
        if next:
            register_tasklet_info(next)
            debugger = get_global_debugger()
            if debugger is not None and next.frame:
                if hasattr(next.frame, 'f_trace'):
                    next.frame.f_trace = debugger.get_thread_local_trace_func()
            debugger = None
        if prev:
            register_tasklet_info(prev)
        try:
            for tasklet_ref, tasklet_info in list(_weak_tasklet_registered_to_info.items()):
                tasklet = tasklet_ref()
                if tasklet is None or not tasklet.alive:
                    try:
                        del _weak_tasklet_registered_to_info[tasklet_ref]
                    except KeyError:
                        pass
                    if tasklet_info.frame_id is not None:
                        remove_custom_frame(tasklet_info.frame_id)
                elif tasklet.paused or tasklet.blocked or tasklet.scheduled:
                    if tasklet.frame and tasklet.frame.f_back:
                        f_back = tasklet.frame.f_back
                        debugger = get_global_debugger()
                        if debugger is not None and debugger.get_file_type(f_back) is None:
                            if tasklet_info.frame_id is None:
                                tasklet_info.frame_id = add_custom_frame(f_back, tasklet_info.tasklet_name, tasklet.thread_id)
                            else:
                                update_custom_frame(tasklet_info.frame_id, f_back, tasklet.thread_id)
                        debugger = None
                elif tasklet.is_current:
                    if tasklet_info.frame_id is not None:
                        remove_custom_frame(tasklet_info.frame_id)
                        tasklet_info.frame_id = None
        finally:
            tasklet = None
            tasklet_info = None
            f_back = None
    except:
        pydev_log.exception()
    if _application_set_schedule_callback is not None:
        return _application_set_schedule_callback(prev, next)