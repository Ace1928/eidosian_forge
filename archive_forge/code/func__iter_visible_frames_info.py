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
def _iter_visible_frames_info(self, py_db, frames_list, flatten_chained=False):
    assert frames_list.__class__ == FramesList
    is_chained = False
    while True:
        for frame in frames_list:
            show_as_current_frame = frame is frames_list.current_frame
            if frame.f_code is None:
                pydev_log.info('Frame without f_code: %s', frame)
                continue
            method_name = frame.f_code.co_name
            if method_name is None:
                pydev_log.info('Frame without co_name: %s', frame)
                continue
            if is_chained:
                method_name = '[Chained Exc: %s] %s' % (frames_list.exc_desc, method_name)
            abs_path_real_path_and_base = get_abs_path_real_path_and_base_from_frame(frame)
            if py_db.get_file_type(frame, abs_path_real_path_and_base) == py_db.PYDEV_FILE:
                frame = frame.f_back
                continue
            frame_id = id(frame)
            lineno = frames_list.frame_id_to_lineno.get(frame_id, frame.f_lineno)
            line_col_info = frames_list.frame_id_to_line_col_info.get(frame_id)
            filename_in_utf8, lineno, changed = py_db.source_mapping.map_to_client(abs_path_real_path_and_base[0], lineno)
            new_filename_in_utf8, applied_mapping = pydevd_file_utils.map_file_to_client(filename_in_utf8)
            applied_mapping = applied_mapping or changed
            yield (frame_id, frame, method_name, abs_path_real_path_and_base[0], new_filename_in_utf8, lineno, applied_mapping, show_as_current_frame, line_col_info)
        if not flatten_chained:
            break
        frames_list = frames_list.chained_frames_list
        if frames_list is None or len(frames_list) == 0:
            break
        is_chained = True