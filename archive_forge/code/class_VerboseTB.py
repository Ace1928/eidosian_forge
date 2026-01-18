from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
class VerboseTB(TBTools):
    """A port of Ka-Ping Yee's cgitb.py module that outputs color text instead
    of HTML.  Requires inspect and pydoc.  Crazy, man.

    Modified version which optionally strips the topmost entries from the
    traceback, to be used with alternate interpreters (because their own code
    would appear in the traceback)."""
    _tb_highlight = 'bg:ansiyellow'
    _tb_highlight_style = 'default'

    def __init__(self, color_scheme: str='Linux', call_pdb: bool=False, ostream=None, tb_offset: int=0, long_header: bool=False, include_vars: bool=True, check_cache=None, debugger_cls=None, parent=None, config=None):
        """Specify traceback offset, headers and color scheme.

        Define how many frames to drop from the tracebacks. Calling it with
        tb_offset=1 allows use of this handler in interpreters which will have
        their own code at the top of the traceback (VerboseTB will first
        remove that frame before printing the traceback info)."""
        TBTools.__init__(self, color_scheme=color_scheme, call_pdb=call_pdb, ostream=ostream, parent=parent, config=config, debugger_cls=debugger_cls)
        self.tb_offset = tb_offset
        self.long_header = long_header
        self.include_vars = include_vars
        if check_cache is None:
            check_cache = linecache.checkcache
        self.check_cache = check_cache
        self.skip_hidden = True

    def format_record(self, frame_info: FrameInfo):
        """Format a single stack frame"""
        assert isinstance(frame_info, FrameInfo)
        Colors = self.Colors
        ColorsNormal = Colors.Normal
        if isinstance(frame_info._sd, stack_data.RepeatedFrames):
            return '    %s[... skipping similar frames: %s]%s\n' % (Colors.excName, frame_info.description, ColorsNormal)
        indent = ' ' * INDENT_SIZE
        em_normal = '%s\n%s%s' % (Colors.valEm, indent, ColorsNormal)
        tpl_call = f'in {Colors.vName}{{file}}{Colors.valEm}{{scope}}{ColorsNormal}'
        tpl_call_fail = 'in %s%%s%s(***failed resolving arguments***)%s' % (Colors.vName, Colors.valEm, ColorsNormal)
        tpl_name_val = '%%s %s= %%s%s' % (Colors.valEm, ColorsNormal)
        link = _format_filename(frame_info.filename, Colors.filenameEm, ColorsNormal, lineno=frame_info.lineno)
        args, varargs, varkw, locals_ = inspect.getargvalues(frame_info.frame)
        if frame_info.executing is not None:
            func = frame_info.executing.code_qualname()
        else:
            func = '?'
        if func == '<module>':
            call = ''
        else:
            var_repr = eqrepr if self.include_vars else nullrepr
            try:
                scope = inspect.formatargvalues(args, varargs, varkw, locals_, formatvalue=var_repr)
                call = tpl_call.format(file=func, scope=scope)
            except KeyError:
                call = tpl_call_fail % func
        lvals = ''
        lvals_list = []
        if self.include_vars:
            try:
                fibp = frame_info.variables_in_executing_piece
                for var in fibp:
                    lvals_list.append(tpl_name_val % (var.name, repr(var.value)))
            except Exception:
                lvals_list.append('Exception trying to inspect frame. No more locals available.')
        if lvals_list:
            lvals = '%s%s' % (indent, em_normal.join(lvals_list))
        result = f'{link}{(', ' if call else '')}{call}\n'
        if frame_info._sd is None:
            tpl_link = '%s%%s%s' % (Colors.filenameEm, ColorsNormal)
            link = tpl_link % util_path.compress_user(frame_info.filename)
            level = '%s %s\n' % (link, call)
            _line_format = PyColorize.Parser(style=self.color_scheme_table.active_scheme_name, parent=self).format2
            first_line = frame_info.code.co_firstlineno
            current_line = frame_info.lineno[0]
            raw_lines = frame_info.raw_lines
            index = current_line - first_line
            if index >= frame_info.context:
                start = max(index - frame_info.context, 0)
                stop = index + frame_info.context
                index = frame_info.context
            else:
                start = 0
                stop = index + frame_info.context
            raw_lines = raw_lines[start:stop]
            return '%s%s' % (level, ''.join(_simple_format_traceback_lines(current_line, index, raw_lines, Colors, lvals, _line_format)))
        else:
            result += ''.join(_format_traceback_lines(frame_info.lines, Colors, self.has_colors, lvals))
        return result

    def prepare_header(self, etype: str, long_version: bool=False):
        colors = self.Colors
        colorsnormal = colors.Normal
        exc = '%s%s%s' % (colors.excName, etype, colorsnormal)
        width = min(75, get_terminal_size()[0])
        if long_version:
            pyver = 'Python ' + sys.version.split()[0] + ': ' + sys.executable
            date = time.ctime(time.time())
            head = '%s%s%s\n%s%s%s\n%s' % (colors.topline, '-' * width, colorsnormal, exc, ' ' * (width - len(etype) - len(pyver)), pyver, date.rjust(width))
            head += '\nA problem occurred executing Python code.  Here is the sequence of function\ncalls leading up to the error, with the most recent (innermost) call last.'
        else:
            head = '%s%s' % (exc, 'Traceback (most recent call last)'.rjust(width - len(etype)))
        return head

    def format_exception(self, etype, evalue):
        colors = self.Colors
        colorsnormal = colors.Normal
        try:
            etype_str, evalue_str = map(str, (etype, evalue))
        except:
            etype, evalue = (str, sys.exc_info()[:2])
            etype_str, evalue_str = map(str, (etype, evalue))
        notes = getattr(evalue, '__notes__', [])
        if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes)):
            notes = [_safe_string(notes, '__notes__', func=repr)]
        return ['{}{}{}: {}'.format(colors.excName, etype_str, colorsnormal, py3compat.cast_unicode(evalue_str)), *('{}{}'.format(colorsnormal, _safe_string(py3compat.cast_unicode(n), 'note')) for n in notes)]

    def format_exception_as_a_whole(self, etype: type, evalue: Optional[BaseException], etb: Optional[TracebackType], number_of_lines_of_context, tb_offset: Optional[int]):
        """Formats the header, traceback and exception message for a single exception.

        This may be called multiple times by Python 3 exception chaining
        (PEP 3134).
        """
        orig_etype = etype
        try:
            etype = etype.__name__
        except AttributeError:
            pass
        tb_offset = self.tb_offset if tb_offset is None else tb_offset
        assert isinstance(tb_offset, int)
        head = self.prepare_header(str(etype), self.long_header)
        records = self.get_records(etb, number_of_lines_of_context, tb_offset) if etb else []
        frames = []
        skipped = 0
        lastrecord = len(records) - 1
        for i, record in enumerate(records):
            if not isinstance(record._sd, stack_data.RepeatedFrames) and self.skip_hidden:
                if record.frame.f_locals.get('__tracebackhide__', 0) and i != lastrecord:
                    skipped += 1
                    continue
            if skipped:
                Colors = self.Colors
                ColorsNormal = Colors.Normal
                frames.append('    %s[... skipping hidden %s frame]%s\n' % (Colors.excName, skipped, ColorsNormal))
                skipped = 0
            frames.append(self.format_record(record))
        if skipped:
            Colors = self.Colors
            ColorsNormal = Colors.Normal
            frames.append('    %s[... skipping hidden %s frame]%s\n' % (Colors.excName, skipped, ColorsNormal))
        formatted_exception = self.format_exception(etype, evalue)
        if records:
            frame_info = records[-1]
            ipinst = get_ipython()
            if ipinst is not None:
                ipinst.hooks.synchronize_with_editor(frame_info.filename, frame_info.lineno, 0)
        return [[head] + frames + formatted_exception]

    def get_records(self, etb: TracebackType, number_of_lines_of_context: int, tb_offset: int):
        assert etb is not None
        context = number_of_lines_of_context - 1
        after = context // 2
        before = context - after
        if self.has_colors:
            style = get_style_by_name(self._tb_highlight_style)
            style = stack_data.style_with_executing_node(style, self._tb_highlight)
            formatter = Terminal256Formatter(style=style)
        else:
            formatter = None
        options = stack_data.Options(before=before, after=after, pygments_formatter=formatter)
        cf: Optional[TracebackType] = etb
        max_len = 0
        tbs = []
        while cf is not None:
            try:
                mod = inspect.getmodule(cf.tb_frame)
                if mod is not None:
                    mod_name = mod.__name__
                    root_name, *_ = mod_name.split('.')
                    if root_name == 'IPython':
                        cf = cf.tb_next
                        continue
                max_len = get_line_number_of_frame(cf.tb_frame)
            except OSError:
                max_len = 0
            max_len = max(max_len, max_len)
            tbs.append(cf)
            cf = getattr(cf, 'tb_next', None)
        if max_len > FAST_THRESHOLD:
            FIs = []
            for tb in tbs:
                frame = tb.tb_frame
                lineno = (frame.f_lineno,)
                code = frame.f_code
                filename = code.co_filename
                FIs.append(FrameInfo('Raw frame', filename, lineno, frame, code, context=context))
            return FIs
        res = list(stack_data.FrameInfo.stack_data(etb, options=options))[tb_offset:]
        res = [FrameInfo._from_stack_data_FrameInfo(r) for r in res]
        return res

    def structured_traceback(self, etype: type, evalue: Optional[BaseException], etb: Optional[TracebackType]=None, tb_offset: Optional[int]=None, number_of_lines_of_context: int=5):
        """Return a nice text document describing the traceback."""
        formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context, tb_offset)
        colors = self.Colors
        colorsnormal = colors.Normal
        head = '%s%s%s' % (colors.topline, '-' * min(75, get_terminal_size()[0]), colorsnormal)
        structured_traceback_parts = [head]
        chained_exceptions_tb_offset = 0
        lines_of_context = 3
        formatted_exceptions = formatted_exception
        exception = self.get_parts_of_chained_exception(evalue)
        if exception:
            assert evalue is not None
            formatted_exceptions += self.prepare_chained_exception_message(evalue.__cause__)
            etype, evalue, etb = exception
        else:
            evalue = None
        chained_exc_ids = set()
        while evalue:
            formatted_exceptions += self.format_exception_as_a_whole(etype, evalue, etb, lines_of_context, chained_exceptions_tb_offset)
            exception = self.get_parts_of_chained_exception(evalue)
            if exception and (not id(exception[1]) in chained_exc_ids):
                chained_exc_ids.add(id(exception[1]))
                formatted_exceptions += self.prepare_chained_exception_message(evalue.__cause__)
                etype, evalue, etb = exception
            else:
                evalue = None
        for formatted_exception in reversed(formatted_exceptions):
            structured_traceback_parts += formatted_exception
        return structured_traceback_parts

    def debugger(self, force: bool=False):
        """Call up the pdb debugger if desired, always clean up the tb
        reference.

        Keywords:

          - force(False): by default, this routine checks the instance call_pdb
            flag and does not actually invoke the debugger if the flag is false.
            The 'force' option forces the debugger to activate even if the flag
            is false.

        If the call_pdb flag is set, the pdb interactive debugger is
        invoked. In all cases, the self.tb reference to the current traceback
        is deleted to prevent lingering references which hamper memory
        management.

        Note that each call to pdb() does an 'import readline', so if your app
        requires a special setup for the readline completers, you'll have to
        fix that by hand after invoking the exception handler."""
        if force or self.call_pdb:
            if self.pdb is None:
                self.pdb = self.debugger_cls()
            display_trap = DisplayTrap(hook=sys.__displayhook__)
            with display_trap:
                self.pdb.reset()
                if hasattr(self, 'tb') and self.tb is not None:
                    etb = self.tb
                else:
                    etb = self.tb = sys.last_traceback
                while self.tb is not None and self.tb.tb_next is not None:
                    assert self.tb.tb_next is not None
                    self.tb = self.tb.tb_next
                if etb and etb.tb_next:
                    etb = etb.tb_next
                self.pdb.botframe = etb.tb_frame
                exc = sys.last_value if sys.version_info < (3, 12) else getattr(sys, 'last_exc', sys.last_value)
                if exc:
                    self.pdb.interaction(None, exc)
                else:
                    self.pdb.interaction(None, etb)
        if hasattr(self, 'tb'):
            del self.tb

    def handler(self, info=None):
        etype, evalue, etb = info or sys.exc_info()
        self.tb = etb
        ostream = self.ostream
        ostream.flush()
        ostream.write(self.text(etype, evalue, etb))
        ostream.write('\n')
        ostream.flush()

    def __call__(self, etype=None, evalue=None, etb=None):
        """This hook can replace sys.excepthook (for Python 2.1 or higher)."""
        if etb is None:
            self.handler()
        else:
            self.handler((etype, evalue, etb))
        try:
            self.debugger()
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt')