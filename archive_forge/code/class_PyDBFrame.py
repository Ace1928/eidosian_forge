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
class PyDBFrame:
    """This makes the tracing for a given frame, so, the trace_dispatch
    is used initially when we enter into a new context ('call') and then
    is reused for the entire context.
    """
    filename_to_lines_where_exceptions_are_ignored = {}
    filename_to_stat_info = {}
    should_skip = -1
    exc_info = ()

    def __init__(self, args):
        self._args = args

    def set_suspend(self, *args, **kwargs):
        self._args[0].set_suspend(*args, **kwargs)

    def do_wait_suspend(self, *args, **kwargs):
        self._args[0].do_wait_suspend(*args, **kwargs)

    def trace_exception(self, frame, event, arg):
        if event == 'exception':
            should_stop, frame = self._should_stop_on_exception(frame, event, arg)
            if should_stop:
                if self._handle_exception(frame, event, arg, EXCEPTION_TYPE_HANDLED):
                    return self.trace_dispatch
        elif event == 'return':
            exc_info = self.exc_info
            if exc_info and arg is None:
                frame_skips_cache, frame_cache_key = (self._args[4], self._args[5])
                custom_key = (frame_cache_key, 'try_exc_info')
                container_obj = frame_skips_cache.get(custom_key)
                if container_obj is None:
                    container_obj = frame_skips_cache[custom_key] = _TryExceptContainerObj()
                if is_unhandled_exception(container_obj, self._args[0], frame, exc_info[1], exc_info[2]) and self.handle_user_exception(frame):
                    return self.trace_dispatch
        return self.trace_exception

    def _should_stop_on_exception(self, frame, event, arg):
        main_debugger = self._args[0]
        info = self._args[2]
        should_stop = False
        if info.pydev_state != 2:
            exception, value, trace = arg
            if trace is not None and hasattr(trace, 'tb_next'):
                should_stop = False
                exception_breakpoint = None
                try:
                    if main_debugger.plugin is not None:
                        result = main_debugger.plugin.exception_break(main_debugger, self, frame, self._args, arg)
                        if result:
                            should_stop, frame = result
                except:
                    pydev_log.exception()
                if not should_stop:
                    if exception == SystemExit and main_debugger.ignore_system_exit_code(value):
                        pass
                    elif exception in (GeneratorExit, StopIteration, StopAsyncIteration):
                        pass
                    elif ignore_exception_trace(trace):
                        pass
                    else:
                        was_just_raised = trace.tb_next is None
                        check_excs = []
                        exc_break_user = main_debugger.get_exception_breakpoint(exception, main_debugger.break_on_user_uncaught_exceptions)
                        if exc_break_user is not None:
                            check_excs.append((exc_break_user, True))
                        exc_break_caught = main_debugger.get_exception_breakpoint(exception, main_debugger.break_on_caught_exceptions)
                        if exc_break_caught is not None:
                            check_excs.append((exc_break_caught, False))
                        for exc_break, is_user_uncaught in check_excs:
                            should_stop = True
                            if main_debugger.exclude_exception_by_filter(exc_break, trace):
                                pydev_log.debug('Ignore exception %s in library %s -- (%s)' % (exception, frame.f_code.co_filename, frame.f_code.co_name))
                                should_stop = False
                            elif exc_break.condition is not None and (not main_debugger.handle_breakpoint_condition(info, exc_break, frame)):
                                should_stop = False
                            elif is_user_uncaught:
                                should_stop = False
                                if not main_debugger.apply_files_filter(frame, frame.f_code.co_filename, True) and (frame.f_back is None or main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True)):
                                    exc_info = self.exc_info
                                    if not exc_info:
                                        exc_info = (arg, frame.f_lineno, set([frame.f_lineno]))
                                    else:
                                        lines = exc_info[2]
                                        lines.add(frame.f_lineno)
                                        exc_info = (arg, frame.f_lineno, lines)
                                    self.exc_info = exc_info
                            elif exc_break.notify_on_first_raise_only and main_debugger.skip_on_exceptions_thrown_in_same_context and (not was_just_raised) and (not just_raised(trace.tb_next)):
                                should_stop = False
                            elif exc_break.notify_on_first_raise_only and (not main_debugger.skip_on_exceptions_thrown_in_same_context) and (not was_just_raised):
                                should_stop = False
                            elif was_just_raised and main_debugger.skip_on_exceptions_thrown_in_same_context:
                                should_stop = False
                            if should_stop:
                                exception_breakpoint = exc_break
                                try:
                                    info.pydev_message = exc_break.qname
                                except:
                                    info.pydev_message = exc_break.qname.encode('utf-8')
                                break
                if should_stop:
                    add_exception_to_frame(frame, (exception, value, trace))
                    if exception_breakpoint is not None and exception_breakpoint.expression is not None:
                        main_debugger.handle_breakpoint_expression(exception_breakpoint, info, frame)
        return (should_stop, frame)

    def handle_user_exception(self, frame):
        exc_info = self.exc_info
        if exc_info:
            return self._handle_exception(frame, 'exception', exc_info[0], EXCEPTION_TYPE_USER_UNHANDLED)
        return False

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

    def get_func_name(self, frame):
        code_obj = frame.f_code
        func_name = code_obj.co_name
        try:
            cls_name = get_clsname_for_code(code_obj, frame)
            if cls_name is not None:
                return '%s.%s' % (cls_name, func_name)
            else:
                return func_name
        except:
            pydev_log.exception()
            return func_name

    def _show_return_values(self, frame, arg):
        try:
            try:
                f_locals_back = getattr(frame.f_back, 'f_locals', None)
                if f_locals_back is not None:
                    return_values_dict = f_locals_back.get(RETURN_VALUES_DICT, None)
                    if return_values_dict is None:
                        return_values_dict = {}
                        f_locals_back[RETURN_VALUES_DICT] = return_values_dict
                    name = self.get_func_name(frame)
                    return_values_dict[name] = arg
            except:
                pydev_log.exception()
        finally:
            f_locals_back = None

    def _remove_return_values(self, main_debugger, frame):
        try:
            try:
                frame.f_locals.pop(RETURN_VALUES_DICT, None)
                f_locals_back = getattr(frame.f_back, 'f_locals', None)
                if f_locals_back is not None:
                    f_locals_back.pop(RETURN_VALUES_DICT, None)
            except:
                pydev_log.exception()
        finally:
            f_locals_back = None

    def _get_unfiltered_back_frame(self, main_debugger, frame):
        f = frame.f_back
        while f is not None:
            if not main_debugger.is_files_filter_enabled:
                return f
            elif main_debugger.apply_files_filter(f, f.f_code.co_filename, False):
                f = f.f_back
            else:
                return f
        return f

    def _is_same_frame(self, target_frame, current_frame):
        if target_frame is current_frame:
            return True
        info = self._args[2]
        if info.pydev_use_scoped_step_frame:
            if target_frame is not None and current_frame is not None:
                if target_frame.f_code.co_filename == current_frame.f_code.co_filename:
                    f = current_frame.f_back
                    if f is not None and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                        f = f.f_back
                        if f is not None and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                            return True
        return False

    def trace_dispatch(self, frame, event, arg):
        try:
            main_debugger, abs_path_canonical_path_and_base, info, thread, frame_skips_cache, frame_cache_key = self._args
            info.is_tracing += 1
            line = frame.f_lineno or 0
            line_cache_key = (frame_cache_key, line)
            if main_debugger.pydb_disposed:
                return None if event == 'call' else NO_FTRACE
            plugin_manager = main_debugger.plugin
            has_exception_breakpoints = main_debugger.break_on_caught_exceptions or main_debugger.break_on_user_uncaught_exceptions or main_debugger.has_plugin_exception_breaks
            stop_frame = info.pydev_step_stop
            step_cmd = info.pydev_step_cmd
            function_breakpoint_on_call_event = None
            if frame.f_code.co_flags & 160:
                if event == 'line':
                    is_line = True
                    is_call = False
                    is_return = False
                    is_exception_event = False
                elif event == 'return':
                    is_line = False
                    is_call = False
                    is_return = True
                    is_exception_event = False
                    returns_cache_key = (frame_cache_key, 'returns')
                    return_lines = frame_skips_cache.get(returns_cache_key)
                    if return_lines is None:
                        return_lines = set()
                        for x in main_debugger.collect_return_info(frame.f_code):
                            return_lines.add(x.return_line)
                        frame_skips_cache[returns_cache_key] = return_lines
                    if line not in return_lines:
                        return self.trace_dispatch
                    else:
                        if self.exc_info:
                            self.handle_user_exception(frame)
                            return self.trace_dispatch
                        if stop_frame is frame and (not info.pydev_use_scoped_step_frame):
                            if step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE):
                                f = self._get_unfiltered_back_frame(main_debugger, frame)
                                if f is not None:
                                    info.pydev_step_cmd = CMD_STEP_INTO_COROUTINE
                                    info.pydev_step_stop = f
                                elif step_cmd == CMD_STEP_OVER:
                                    info.pydev_step_cmd = CMD_STEP_INTO
                                    info.pydev_step_stop = None
                                elif step_cmd == CMD_STEP_OVER_MY_CODE:
                                    info.pydev_step_cmd = CMD_STEP_INTO_MY_CODE
                                    info.pydev_step_stop = None
                            elif step_cmd == CMD_STEP_INTO_COROUTINE:
                                f = self._get_unfiltered_back_frame(main_debugger, frame)
                                if f is not None:
                                    info.pydev_step_stop = f
                                else:
                                    info.pydev_step_cmd = CMD_STEP_INTO
                                    info.pydev_step_stop = None
                elif event == 'exception':
                    breakpoints_for_file = None
                    if has_exception_breakpoints:
                        should_stop, frame = self._should_stop_on_exception(frame, event, arg)
                        if should_stop:
                            if self._handle_exception(frame, event, arg, EXCEPTION_TYPE_HANDLED):
                                return self.trace_dispatch
                    return self.trace_dispatch
                else:
                    return self.trace_dispatch
            elif event == 'line':
                is_line = True
                is_call = False
                is_return = False
                is_exception_event = False
            elif event == 'return':
                is_line = False
                is_return = True
                is_call = False
                is_exception_event = False
                if stop_frame is frame and (not info.pydev_use_scoped_step_frame) and is_return and (step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_STEP_OVER_MY_CODE, CMD_STEP_RETURN_MY_CODE, CMD_SMART_STEP_INTO)):
                    if step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_SMART_STEP_INTO):
                        info.pydev_step_cmd = CMD_STEP_INTO
                    else:
                        info.pydev_step_cmd = CMD_STEP_INTO_MY_CODE
                    info.pydev_step_stop = None
                if self.exc_info:
                    if self.handle_user_exception(frame):
                        return self.trace_dispatch
            elif event == 'call':
                is_line = False
                is_call = True
                is_return = False
                is_exception_event = False
                if frame.f_code.co_firstlineno == frame.f_lineno:
                    function_breakpoint_on_call_event = main_debugger.function_breakpoint_name_to_breakpoint.get(frame.f_code.co_name)
            elif event == 'exception':
                is_exception_event = True
                breakpoints_for_file = None
                if has_exception_breakpoints:
                    should_stop, frame = self._should_stop_on_exception(frame, event, arg)
                    if should_stop:
                        if self._handle_exception(frame, event, arg, EXCEPTION_TYPE_HANDLED):
                            return self.trace_dispatch
                is_line = False
                is_return = False
                is_call = False
            else:
                return self.trace_dispatch
            if not is_exception_event:
                breakpoints_for_file = main_debugger.breakpoints.get(abs_path_canonical_path_and_base[1])
                can_skip = False
                if info.pydev_state == 1:
                    if step_cmd == -1:
                        can_skip = True
                    elif step_cmd in (CMD_STEP_OVER, CMD_STEP_RETURN, CMD_STEP_OVER_MY_CODE, CMD_STEP_RETURN_MY_CODE) and (not self._is_same_frame(stop_frame, frame)):
                        can_skip = True
                    elif step_cmd == CMD_SMART_STEP_INTO and (stop_frame is not None and stop_frame is not frame and (stop_frame is not frame.f_back) and (frame.f_back is None or stop_frame is not frame.f_back.f_back)):
                        can_skip = True
                    elif step_cmd == CMD_STEP_INTO_MY_CODE:
                        if main_debugger.apply_files_filter(frame, frame.f_code.co_filename, True) and (frame.f_back is None or main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True)):
                            can_skip = True
                    elif step_cmd == CMD_STEP_INTO_COROUTINE:
                        f = frame
                        while f is not None:
                            if self._is_same_frame(stop_frame, f):
                                break
                            f = f.f_back
                        else:
                            can_skip = True
                    if can_skip:
                        if plugin_manager is not None and (main_debugger.has_plugin_line_breaks or main_debugger.has_plugin_exception_breaks):
                            can_skip = plugin_manager.can_skip(main_debugger, frame)
                        if can_skip and main_debugger.show_return_values and (info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE)) and self._is_same_frame(stop_frame, frame.f_back):
                            can_skip = False
                if function_breakpoint_on_call_event:
                    pass
                elif not breakpoints_for_file:
                    if can_skip:
                        if has_exception_breakpoints:
                            return self.trace_exception
                        else:
                            return None if is_call else NO_FTRACE
                else:
                    if can_skip:
                        breakpoints_in_line_cache = frame_skips_cache.get(line_cache_key, -1)
                        if breakpoints_in_line_cache == 0:
                            return self.trace_dispatch
                    breakpoints_in_frame_cache = frame_skips_cache.get(frame_cache_key, -1)
                    if breakpoints_in_frame_cache != -1:
                        has_breakpoint_in_frame = breakpoints_in_frame_cache == 1
                    else:
                        has_breakpoint_in_frame = False
                        try:
                            func_lines = set()
                            for offset_and_lineno in dis.findlinestarts(frame.f_code):
                                func_lines.add(offset_and_lineno[1])
                        except:
                            curr_func_name = frame.f_code.co_name
                            if curr_func_name in ('?', '<module>', '<lambda>'):
                                curr_func_name = ''
                            for bp in breakpoints_for_file.values():
                                if bp.func_name in ('None', curr_func_name):
                                    has_breakpoint_in_frame = True
                                    break
                        else:
                            for bp_line in breakpoints_for_file:
                                if bp_line in func_lines:
                                    has_breakpoint_in_frame = True
                                    break
                        if has_breakpoint_in_frame:
                            frame_skips_cache[frame_cache_key] = 1
                        else:
                            frame_skips_cache[frame_cache_key] = 0
                    if can_skip and (not has_breakpoint_in_frame):
                        if has_exception_breakpoints:
                            return self.trace_exception
                        else:
                            return None if is_call else NO_FTRACE
            try:
                stop_on_plugin_breakpoint = False
                stop_info = {}
                breakpoint = None
                stop = False
                stop_reason = CMD_SET_BREAK
                bp_type = None
                if function_breakpoint_on_call_event:
                    breakpoint = function_breakpoint_on_call_event
                    stop = True
                    new_frame = frame
                    stop_reason = CMD_SET_FUNCTION_BREAK
                elif is_line and info.pydev_state != STATE_SUSPEND and (breakpoints_for_file is not None) and (line in breakpoints_for_file):
                    breakpoint = breakpoints_for_file[line]
                    new_frame = frame
                    stop = True
                elif plugin_manager is not None and main_debugger.has_plugin_line_breaks:
                    result = plugin_manager.get_breakpoint(main_debugger, self, frame, event, self._args)
                    if result:
                        stop_on_plugin_breakpoint, breakpoint, new_frame, bp_type = result
                if breakpoint:
                    if breakpoint.expression is not None:
                        main_debugger.handle_breakpoint_expression(breakpoint, info, new_frame)
                    if stop or stop_on_plugin_breakpoint:
                        eval_result = False
                        if breakpoint.has_condition:
                            eval_result = main_debugger.handle_breakpoint_condition(info, breakpoint, new_frame)
                            if not eval_result:
                                stop = False
                                stop_on_plugin_breakpoint = False
                    if is_call and (frame.f_code.co_name in ('<lambda>', '<module>') or (line == 1 and frame.f_code.co_name.startswith('<cell'))):
                        return self.trace_dispatch
                    if (stop or stop_on_plugin_breakpoint) and breakpoint.is_logpoint:
                        stop = False
                        stop_on_plugin_breakpoint = False
                        if info.pydev_message is not None and len(info.pydev_message) > 0:
                            cmd = main_debugger.cmd_factory.make_io_message(info.pydev_message + os.linesep, '1')
                            main_debugger.writer.add_command(cmd)
                if main_debugger.show_return_values:
                    if is_return and (info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_SMART_STEP_INTO) and self._is_same_frame(stop_frame, frame.f_back) or (info.pydev_step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE) and self._is_same_frame(stop_frame, frame)) or info.pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_COROUTINE) or (info.pydev_step_cmd == CMD_STEP_INTO_MY_CODE and frame.f_back is not None and (not main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, True)))):
                        self._show_return_values(frame, arg)
                elif main_debugger.remove_return_values_flag:
                    try:
                        self._remove_return_values(main_debugger, frame)
                    finally:
                        main_debugger.remove_return_values_flag = False
                if stop:
                    self.set_suspend(thread, stop_reason, suspend_other_threads=breakpoint and breakpoint.suspend_policy == 'ALL')
                elif stop_on_plugin_breakpoint and plugin_manager is not None:
                    result = plugin_manager.suspend(main_debugger, thread, frame, bp_type)
                    if result:
                        frame = result
                if info.pydev_state == STATE_SUSPEND:
                    self.do_wait_suspend(thread, frame, event, arg)
                    return self.trace_dispatch
                elif not breakpoint and is_line:
                    frame_skips_cache[line_cache_key] = 0
            except:
                exc = sys.exc_info()[0]
                cmd = main_debugger.cmd_factory.make_console_message('%s raised from within the callback set in sys.settrace.\nDebugging will be disabled for this thread (%s).\n' % (exc, thread))
                main_debugger.writer.add_command(cmd)
                if not issubclass(exc, (KeyboardInterrupt, SystemExit)):
                    pydev_log.exception()
                raise
            try:
                should_skip = 0
                if pydevd_dont_trace.should_trace_hook is not None:
                    if self.should_skip == -1:
                        if not pydevd_dont_trace.should_trace_hook(frame, abs_path_canonical_path_and_base[0]):
                            should_skip = self.should_skip = 1
                        else:
                            should_skip = self.should_skip = 0
                    else:
                        should_skip = self.should_skip
                plugin_stop = False
                if should_skip:
                    stop = False
                elif step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE, CMD_STEP_INTO_COROUTINE):
                    force_check_project_scope = step_cmd == CMD_STEP_INTO_MY_CODE
                    if is_line:
                        if not info.pydev_use_scoped_step_frame:
                            if force_check_project_scope or main_debugger.is_files_filter_enabled:
                                stop = not main_debugger.apply_files_filter(frame, frame.f_code.co_filename, force_check_project_scope)
                            else:
                                stop = True
                        else:
                            if force_check_project_scope or main_debugger.is_files_filter_enabled:
                                if not not main_debugger.apply_files_filter(frame, frame.f_code.co_filename, force_check_project_scope):
                                    return None if is_call else NO_FTRACE
                            filename = frame.f_code.co_filename
                            if filename.endswith('.pyc'):
                                filename = filename[:-1]
                            if not filename.endswith(PYDEVD_IPYTHON_CONTEXT[0]):
                                f = frame.f_back
                                while f is not None:
                                    if f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                                        f2 = f.f_back
                                        if f2 is not None and f2.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                                            pydev_log.debug('Stop inside ipython call')
                                            stop = True
                                            break
                                    f = f.f_back
                                del f
                            if not stop:
                                return None if is_call else NO_FTRACE
                    elif is_return and frame.f_back is not None and (not info.pydev_use_scoped_step_frame):
                        if main_debugger.get_file_type(frame.f_back) == main_debugger.PYDEV_FILE:
                            stop = False
                        elif force_check_project_scope or main_debugger.is_files_filter_enabled:
                            stop = not main_debugger.apply_files_filter(frame.f_back, frame.f_back.f_code.co_filename, force_check_project_scope)
                            if stop:
                                if info.step_in_initial_location == (frame.f_back, frame.f_back.f_lineno):
                                    stop = False
                        else:
                            stop = True
                    else:
                        stop = False
                    if stop:
                        if step_cmd == CMD_STEP_INTO_COROUTINE:
                            f = frame
                            while f is not None:
                                if self._is_same_frame(stop_frame, f):
                                    break
                                f = f.f_back
                            else:
                                stop = False
                    if plugin_manager is not None:
                        result = plugin_manager.cmd_step_into(main_debugger, frame, event, self._args, stop_info, stop)
                        if result:
                            stop, plugin_stop = result
                elif step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE):
                    stop = self._is_same_frame(stop_frame, frame) and is_line
                    if plugin_manager is not None:
                        result = plugin_manager.cmd_step_over(main_debugger, frame, event, self._args, stop_info, stop)
                        if result:
                            stop, plugin_stop = result
                elif step_cmd == CMD_SMART_STEP_INTO:
                    stop = False
                    back = frame.f_back
                    if self._is_same_frame(stop_frame, frame) and is_return:
                        stop = True
                    elif self._is_same_frame(stop_frame, back) and is_line:
                        if info.pydev_smart_child_offset != -1:
                            stop = False
                        else:
                            pydev_smart_parent_offset = info.pydev_smart_parent_offset
                            pydev_smart_step_into_variants = info.pydev_smart_step_into_variants
                            if pydev_smart_parent_offset >= 0 and pydev_smart_step_into_variants:
                                stop = get_smart_step_into_variant_from_frame_offset(back.f_lasti, pydev_smart_step_into_variants) is get_smart_step_into_variant_from_frame_offset(pydev_smart_parent_offset, pydev_smart_step_into_variants)
                            else:
                                curr_func_name = frame.f_code.co_name
                                if curr_func_name in ('?', '<module>') or curr_func_name is None:
                                    curr_func_name = ''
                                if curr_func_name == info.pydev_func_name and stop_frame.f_lineno == info.pydev_next_line:
                                    stop = True
                        if not stop:
                            return None if is_call else NO_FTRACE
                    elif back is not None and self._is_same_frame(stop_frame, back.f_back) and is_line:
                        pydev_smart_parent_offset = info.pydev_smart_parent_offset
                        pydev_smart_child_offset = info.pydev_smart_child_offset
                        stop = False
                        if pydev_smart_child_offset >= 0 and pydev_smart_child_offset >= 0:
                            pydev_smart_step_into_variants = info.pydev_smart_step_into_variants
                            if pydev_smart_parent_offset >= 0 and pydev_smart_step_into_variants:
                                smart_step_into_variant = get_smart_step_into_variant_from_frame_offset(pydev_smart_parent_offset, pydev_smart_step_into_variants)
                                children_variants = smart_step_into_variant.children_variants
                                stop = children_variants and get_smart_step_into_variant_from_frame_offset(back.f_lasti, children_variants) is get_smart_step_into_variant_from_frame_offset(pydev_smart_child_offset, children_variants)
                        if not stop:
                            return None if is_call else NO_FTRACE
                elif step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
                    stop = is_return and self._is_same_frame(stop_frame, frame)
                else:
                    stop = False
                if stop and step_cmd != -1 and is_return and hasattr(frame, 'f_back'):
                    f_code = getattr(frame.f_back, 'f_code', None)
                    if f_code is not None:
                        if main_debugger.get_file_type(frame.f_back) == main_debugger.PYDEV_FILE:
                            stop = False
                if plugin_stop:
                    stopped_on_plugin = plugin_manager.stop(main_debugger, frame, event, self._args, stop_info, arg, step_cmd)
                elif stop:
                    if is_line:
                        self.set_suspend(thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
                        self.do_wait_suspend(thread, frame, event, arg)
                    elif is_return:
                        back = frame.f_back
                        if back is not None:
                            back_absolute_filename, _, base = get_abs_path_real_path_and_base_from_frame(back)
                            if (base, back.f_code.co_name) in (DEBUG_START, DEBUG_START_PY3K):
                                back = None
                            elif base == TRACE_PROPERTY:
                                return None if is_call else NO_FTRACE
                            elif pydevd_dont_trace.should_trace_hook is not None:
                                if not pydevd_dont_trace.should_trace_hook(back, back_absolute_filename):
                                    main_debugger.set_trace_for_frame_and_parents(back)
                                    return None if is_call else NO_FTRACE
                        if back is not None:
                            self.set_suspend(thread, step_cmd, original_step_cmd=info.pydev_original_step_cmd)
                            self.do_wait_suspend(thread, back, event, arg)
                        else:
                            info.pydev_step_stop = None
                            info.pydev_original_step_cmd = -1
                            info.pydev_step_cmd = -1
                            info.pydev_state = STATE_RUN
                if main_debugger.quitting:
                    return None if is_call else NO_FTRACE
                return self.trace_dispatch
            except:
                exc = sys.exc_info()[0]
                cmd = main_debugger.cmd_factory.make_console_message('%s raised from within the callback set in sys.settrace.\nDebugging will be disabled for this thread (%s).\n' % (exc, thread))
                main_debugger.writer.add_command(cmd)
                if not issubclass(exc, (KeyboardInterrupt, SystemExit)):
                    pydev_log.exception()
                raise
        finally:
            info.is_tracing -= 1