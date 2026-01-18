import json
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle._pydev_saved_modules import thread
from _pydevd_bundle import pydevd_xml, pydevd_frame_utils, pydevd_constants, pydevd_utils
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, get_thread_id,
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND, NULL_EXIT_COMMAND
from _pydevd_bundle.pydevd_utils import quote_smart as quote, get_non_pydevd_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
import pydevd_file_utils
from pydevd_tracing import get_exception_traceback_str
from _pydev_bundle._pydev_completer import completions_to_xml
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_frame_utils import FramesList
from io import StringIO
def _make_send_curr_exception_trace_str(self, py_db, thread_id, exc_type, exc_desc, trace_obj):
    frames_list = pydevd_frame_utils.create_frames_list_from_traceback(trace_obj, None, exc_type, exc_desc)
    exc_type = pydevd_xml.make_valid_xml_value(str(exc_type)).replace('\t', '  ') or 'exception: type unknown'
    exc_desc = pydevd_xml.make_valid_xml_value(str(exc_desc)).replace('\t', '  ') or 'exception: no description'
    thread_suspend_str, thread_stack_str = self.make_thread_suspend_str(py_db, thread_id, frames_list, CMD_SEND_CURR_EXCEPTION_TRACE, '')
    return (exc_type, exc_desc, thread_suspend_str, thread_stack_str)