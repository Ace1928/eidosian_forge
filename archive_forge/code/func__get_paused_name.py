import pydevd_tracing
import greenlet
import gevent
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import add_custom_frame, update_custom_frame, remove_custom_frame
from _pydevd_bundle.pydevd_constants import GEVENT_SHOW_PAUSED_GREENLETS, get_global_debugger, \
from _pydev_bundle import pydev_log
from pydevd_file_utils import basename
def _get_paused_name(py_db, g):
    frame = g.gr_frame
    use_frame = frame
    while use_frame is not None:
        if py_db.apply_files_filter(use_frame, use_frame.f_code.co_filename, True):
            frame = use_frame
            use_frame = use_frame.f_back
        else:
            break
    if use_frame is None:
        use_frame = frame
    return '%s: %s - %s' % (type(g).__name__, use_frame.f_code.co_name, basename(use_frame.f_code.co_filename))