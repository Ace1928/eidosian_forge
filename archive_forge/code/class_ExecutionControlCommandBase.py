from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class ExecutionControlCommandBase(gdb.Command):
    """
    Superclass for language specific execution control. Language specific
    features should be implemented by lang_info using the LanguageInfo
    interface. 'name' is the name of the command.
    """

    def __init__(self, name, lang_info):
        super(ExecutionControlCommandBase, self).__init__(name, gdb.COMMAND_RUNNING, gdb.COMPLETE_NONE)
        self.lang_info = lang_info

    def install_breakpoints(self):
        all_locations = itertools.chain(self.lang_info.static_break_functions(), self.lang_info.runtime_break_functions())
        for location in all_locations:
            result = gdb.execute('break %s' % location, to_string=True)
            yield re.search('Breakpoint (\\d+)', result).group(1)

    def delete_breakpoints(self, breakpoint_list):
        for bp in breakpoint_list:
            gdb.execute('delete %s' % bp)

    def filter_output(self, result):
        reflags = re.MULTILINE
        output_on_halt = [('^Program received signal .*', reflags | re.DOTALL), ('.*[Ww]arning.*', 0), ('^Program exited .*', reflags)]
        output_always = [('^(Old|New) value = .*', reflags), ('^\\d+: \\w+ = .*', reflags)]

        def filter_output(regexes):
            output = []
            for regex, flags in regexes:
                for match in re.finditer(regex, result, flags):
                    output.append(match.group(0))
            return '\n'.join(output)
        match_finish = re.search('^Value returned is \\$\\d+ = (.*)', result, re.MULTILINE)
        if match_finish:
            finish_output = 'Value returned: %s\n' % match_finish.group(1)
        else:
            finish_output = ''
        return (filter_output(output_on_halt), finish_output + filter_output(output_always))

    def stopped(self):
        return get_selected_inferior().pid == 0

    def finish_executing(self, result):
        """
        After doing some kind of code running in the inferior, print the line
        of source code or the result of the last executed gdb command (passed
        in as the `result` argument).
        """
        output_on_halt, output_always = self.filter_output(result)
        if self.stopped():
            print(output_always)
            print(output_on_halt)
        else:
            frame = gdb.selected_frame()
            source_line = self.lang_info.get_source_line(frame)
            if self.lang_info.is_relevant_function(frame):
                raised_exception = self.lang_info.exc_info(frame)
                if raised_exception:
                    print(raised_exception)
            if source_line:
                if output_always.rstrip():
                    print(output_always.rstrip())
                print(source_line)
            else:
                print(result)

    def _finish(self):
        """
        Execute until the function returns (or until something else makes it
        stop)
        """
        if gdb.selected_frame().older() is not None:
            return gdb.execute('finish', to_string=True)
        else:
            return gdb.execute('cont', to_string=True)

    def _finish_frame(self):
        """
        Execute until the function returns to a relevant caller.
        """
        while True:
            result = self._finish()
            try:
                frame = gdb.selected_frame()
            except RuntimeError:
                break
            hitbp = re.search('Breakpoint (\\d+)', result)
            is_relevant = self.lang_info.is_relevant_function(frame)
            if hitbp or is_relevant or self.stopped():
                break
        return result

    def finish(self, *args):
        """Implements the finish command."""
        result = self._finish_frame()
        self.finish_executing(result)

    def step(self, stepinto, stepover_command='next'):
        """
        Do a single step or step-over. Returns the result of the last gdb
        command that made execution stop.

        This implementation, for stepping, sets (conditional) breakpoints for
        all functions that are deemed relevant. It then does a step over until
        either something halts execution, or until the next line is reached.

        If, however, stepover_command is given, it should be a string gdb
        command that continues execution in some way. The idea is that the
        caller has set a (conditional) breakpoint or watchpoint that can work
        more efficiently than the step-over loop. For Python this means setting
        a watchpoint for f->f_lasti, which means we can then subsequently
        "finish" frames.
        We want f->f_lasti instead of f->f_lineno, because the latter only
        works properly with local trace functions, see
        PyFrameObjectPtr.current_line_num and PyFrameObjectPtr.addr2line.
        """
        if stepinto:
            breakpoint_list = list(self.install_breakpoints())
        beginframe = gdb.selected_frame()
        if self.lang_info.is_relevant_function(beginframe):
            beginline = self.lang_info.lineno(beginframe)
            if not stepinto:
                depth = stackdepth(beginframe)
        newframe = beginframe
        while True:
            if self.lang_info.is_relevant_function(newframe):
                result = gdb.execute(stepover_command, to_string=True)
            else:
                result = self._finish_frame()
            if self.stopped():
                break
            newframe = gdb.selected_frame()
            is_relevant_function = self.lang_info.is_relevant_function(newframe)
            try:
                framename = newframe.name()
            except RuntimeError:
                framename = None
            m = re.search('Breakpoint (\\d+)', result)
            if m:
                if is_relevant_function and m.group(1) in breakpoint_list:
                    break
            if newframe != beginframe:
                if not stepinto:
                    newdepth = stackdepth(newframe)
                    is_relevant_function = newdepth < depth and is_relevant_function
                if is_relevant_function:
                    break
            else:
                lineno = self.lang_info.lineno(newframe)
                if lineno and lineno != beginline:
                    break
        if stepinto:
            self.delete_breakpoints(breakpoint_list)
        self.finish_executing(result)

    def run(self, args, from_tty):
        self.finish_executing(gdb.execute('run ' + args, to_string=True))

    def cont(self, *args):
        self.finish_executing(gdb.execute('cont', to_string=True))