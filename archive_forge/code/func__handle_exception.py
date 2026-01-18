import linecache
import os.path
import re
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (RETURN_VALUES_DICT, NO_FTRACE,
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, just_raised, remove_exception_from_frame, ignore_exception_trace
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydevd_bundle.pydevd_comm_constants import constant_to_str, CMD_SET_FUNCTION_BREAK
import sys
import dis
def _handle_exception(self, frame, event, arg, exception_type):
    stopped = False
    try:
        trace_obj = arg[2]
        main_debugger = self._args[0]
        initial_trace_obj = trace_obj
        if trace_obj.tb_next is None and trace_obj.tb_frame is frame:
            pass
        else:
            while trace_obj.tb_next is not None:
                trace_obj = trace_obj.tb_next
        if main_debugger.ignore_exceptions_thrown_in_lines_with_ignore_exception:
            for check_trace_obj in (initial_trace_obj, trace_obj):
                abs_real_path_and_base = get_abs_path_real_path_and_base_from_frame(check_trace_obj.tb_frame)
                absolute_filename = abs_real_path_and_base[0]
                canonical_normalized_filename = abs_real_path_and_base[1]
                filename_to_lines_where_exceptions_are_ignored = self.filename_to_lines_where_exceptions_are_ignored
                lines_ignored = filename_to_lines_where_exceptions_are_ignored.get(canonical_normalized_filename)
                if lines_ignored is None:
                    lines_ignored = filename_to_lines_where_exceptions_are_ignored[canonical_normalized_filename] = {}
                try:
                    curr_stat = os.stat(absolute_filename)
                    curr_stat = (curr_stat.st_size, curr_stat.st_mtime)
                except:
                    curr_stat = None
                last_stat = self.filename_to_stat_info.get(absolute_filename)
                if last_stat != curr_stat:
                    self.filename_to_stat_info[absolute_filename] = curr_stat
                    lines_ignored.clear()
                    try:
                        linecache.checkcache(absolute_filename)
                    except:
                        pydev_log.exception('Error in linecache.checkcache(%r)', absolute_filename)
                from_user_input = main_debugger.filename_to_lines_where_exceptions_are_ignored.get(canonical_normalized_filename)
                if from_user_input:
                    merged = {}
                    merged.update(lines_ignored)
                    merged.update(from_user_input)
                else:
                    merged = lines_ignored
                exc_lineno = check_trace_obj.tb_lineno
                if exc_lineno not in merged:
                    try:
                        line = linecache.getline(absolute_filename, exc_lineno, check_trace_obj.tb_frame.f_globals)
                    except:
                        pydev_log.exception('Error in linecache.getline(%r, %s, f_globals)', absolute_filename, exc_lineno)
                        line = ''
                    if IGNORE_EXCEPTION_TAG.match(line) is not None:
                        lines_ignored[exc_lineno] = 1
                        return False
                    else:
                        lines_ignored[exc_lineno] = 0
                elif merged.get(exc_lineno, 0):
                    return False
        thread = self._args[3]
        try:
            frame_id_to_frame = {}
            frame_id_to_frame[id(frame)] = frame
            f = trace_obj.tb_frame
            while f is not None:
                frame_id_to_frame[id(f)] = f
                f = f.f_back
            f = None
            stopped = True
            main_debugger.send_caught_exception_stack(thread, arg, id(frame))
            try:
                self.set_suspend(thread, CMD_STEP_CAUGHT_EXCEPTION)
                self.do_wait_suspend(thread, frame, event, arg, exception_type=exception_type)
            finally:
                main_debugger.send_caught_exception_stack_proceeded(thread)
        except:
            pydev_log.exception()
        main_debugger.set_trace_for_frame_and_parents(frame)
    finally:
        remove_exception_from_frame(frame)
        frame = None
        trace_obj = None
        initial_trace_obj = None
        check_trace_obj = None
        f = None
        frame_id_to_frame = None
        main_debugger = None
        thread = None
    return stopped