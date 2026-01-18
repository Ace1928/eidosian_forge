import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
class Repl(metaclass=abc.ABCMeta):
    """Implements the necessary guff for a Python-repl-alike interface

    The execution of the code entered and all that stuff was taken from the
    Python code module, I had to copy it instead of inheriting it, I can't
    remember why. The rest of the stuff is basically what makes it fancy.

    It reads what you type, passes it to a lexer and highlighter which
    returns a formatted string. This then gets passed to echo() which
    parses that string and prints to the curses screen in appropriate
    colours and/or bold attribute.

    The Repl class also keeps two stacks of lines that the user has typed in:
    One to be used for the undo feature. I am not happy with the way this
    works.  The only way I have been able to think of is to keep the code
    that's been typed in in memory and re-evaluate it in its entirety for each
    "undo" operation. Obviously this means some operations could be extremely
    slow.  I'm not even by any means certain that this truly represents a
    genuine "undo" implementation, but it does seem to be generally pretty
    effective.

    If anyone has any suggestions for how this could be improved, I'd be happy
    to hear them and implement it/accept a patch. I researched a bit into the
    idea of keeping the entire Python state in memory, but this really seems
    very difficult (I believe it may actually be impossible to work) and has
    its own problems too.

    The other stack is for keeping a history for pressing the up/down keys
    to go back and forth between lines.

    XXX Subclasses should implement echo, current_line, cw
    """

    @abc.abstractmethod
    def reevaluate(self):
        pass

    @abc.abstractmethod
    def reprint_line(self, lineno: int, tokens: List[Tuple[_TokenType, str]]) -> None:
        pass

    @abc.abstractmethod
    def _get_current_line(self) -> str:
        pass

    @abc.abstractmethod
    def _set_current_line(self, val: str) -> None:
        pass

    @property
    def current_line(self) -> str:
        """The current line"""
        return self._get_current_line()

    @current_line.setter
    def current_line(self, value: str) -> None:
        self._set_current_line(value)

    @abc.abstractmethod
    def _get_cursor_offset(self) -> int:
        pass

    @abc.abstractmethod
    def _set_cursor_offset(self, val: int) -> None:
        pass

    @property
    def cursor_offset(self) -> int:
        """The current cursor offset from the front of the "line"."""
        return self._get_cursor_offset()

    @cursor_offset.setter
    def cursor_offset(self, value: int) -> None:
        self._set_cursor_offset(value)
    if TYPE_CHECKING:
        cpos: int

    def __init__(self, interp: Interpreter, config: Config):
        """Initialise the repl.

        interp is a Python code.InteractiveInterpreter instance

        config is a populated bpython.config.Struct.
        """
        self.config = config
        self.cut_buffer = ''
        self.buffer: List[str] = []
        self.interp = interp
        self.interp.syntaxerror_callback = self.clear_current_line
        self.match = False
        self.rl_history = History(duplicates=config.hist_duplicates, hist_size=config.hist_length)
        self.screen_hist: List[str] = []
        self.history: List[str] = []
        self.redo_stack: List[str] = []
        self.evaluating = False
        self.matches_iter = MatchesIterator()
        self.funcprops = None
        self.arg_pos: Union[str, int, None] = None
        self.current_func = None
        self.highlighted_paren: Optional[Tuple[Any, List[Tuple[_TokenType, str]]]] = None
        self._C: Dict[str, int] = {}
        self.prev_block_finished: int = 0
        self.interact: Interaction = NoInteraction(self.config)
        self.prev_pastebin_content = ''
        self.prev_pastebin_url = ''
        self.prev_removal_url = ''
        self.closed = False
        self.paster: Union[PasteHelper, PastePinnwand]
        if self.config.hist_file.exists():
            try:
                self.rl_history.load(self.config.hist_file, getpreferredencoding() or 'ascii')
            except OSError:
                pass
        self.module_gatherer = ModuleGatherer(skiplist=self.config.import_completion_skiplist)
        self.completers = autocomplete.get_default_completer(config.autocomplete_mode, self.module_gatherer)
        if self.config.pastebin_helper:
            self.paster = PasteHelper(self.config.pastebin_helper)
        else:
            self.paster = PastePinnwand(self.config.pastebin_url, self.config.pastebin_expiry)

    @property
    def ps1(self) -> str:
        return cast(str, getattr(sys, 'ps1', '>>> '))

    @property
    def ps2(self) -> str:
        return cast(str, getattr(sys, 'ps2', '... '))

    def startup(self) -> None:
        """
        Execute PYTHONSTARTUP file if it exits. Call this after front
        end-specific initialisation.
        """
        filename = os.environ.get('PYTHONSTARTUP')
        if filename:
            encoding = inspection.get_encoding_file(filename)
            with open(filename, encoding=encoding) as f:
                source = f.read()
                self.interp.runsource(source, filename, 'exec')

    def current_string(self, concatenate=False):
        """If the line ends in a string get it, otherwise return ''"""
        tokens = self.tokenize(self.current_line)
        string_tokens = list(takewhile(token_is_any_of([Token.String, Token.Text]), reversed(tokens)))
        if not string_tokens:
            return ''
        opening = string_tokens.pop()[1]
        string = list()
        for token, value in reversed(string_tokens):
            if token is Token.Text:
                continue
            elif opening is None:
                opening = value
            elif token is Token.String.Doc:
                string.append(value[3:-3])
                opening = None
            elif value == opening:
                opening = None
                if not concatenate:
                    string = list()
            else:
                string.append(value)
        if opening is None:
            return ''
        return ''.join(string)

    def get_object(self, name: str) -> Any:
        attributes = name.split('.')
        obj = eval(attributes.pop(0), cast(Dict[str, Any], self.interp.locals))
        while attributes:
            obj = inspection.getattr_safe(obj, attributes.pop(0))
        return obj

    @classmethod
    def _funcname_and_argnum(cls, line: str) -> Tuple[Optional[str], Optional[Union[str, int]]]:
        """Parse out the current function name and arg from a line of code."""
        stack = [_FuncExpr('', '', 0, '')]
        try:
            for token, value in Python3Lexer().get_tokens(line):
                if token is Token.Punctuation:
                    if value in '([{':
                        stack.append(_FuncExpr('', '', 0, value))
                    elif value in ')]}':
                        element = stack.pop()
                        expr = element.opening + element.full_expr + value
                        stack[-1].function_expr += expr
                        stack[-1].full_expr += expr
                    elif value == ',':
                        if stack[-1].keyword is None:
                            stack[-1].arg_number += 1
                        else:
                            stack[-1].keyword = ''
                        stack[-1].function_expr = ''
                        stack[-1].full_expr += value
                    elif value == ':' and stack[-1].opening == 'lambda':
                        expr = stack.pop().full_expr + ':'
                        stack[-1].function_expr += expr
                        stack[-1].full_expr += expr
                    else:
                        stack[-1].function_expr = ''
                        stack[-1].full_expr += value
                elif token is Token.Number or token in Token.Number.subtypes or token is Token.Name or (token in Token.Name.subtypes) or (token is Token.Operator and value == '.'):
                    stack[-1].function_expr += value
                    stack[-1].full_expr += value
                elif token is Token.Operator and value == '=':
                    stack[-1].keyword = stack[-1].function_expr
                    stack[-1].function_expr = ''
                    stack[-1].full_expr += value
                elif token is Token.Number or token in Token.Number.subtypes:
                    stack[-1].function_expr = value
                    stack[-1].full_expr += value
                elif token is Token.Keyword and value == 'lambda':
                    stack.append(_FuncExpr(value, '', 0, value))
                else:
                    stack[-1].function_expr = ''
                    stack[-1].full_expr += value
            while stack[-1].opening in '[{':
                stack.pop()
            elem1 = stack.pop()
            elem2 = stack.pop()
            return (elem2.function_expr, elem1.keyword or elem1.arg_number)
        except IndexError:
            return (None, None)

    def get_args(self):
        """Check if an unclosed parenthesis exists, then attempt to get the
        argspec() for it. On success, update self.funcprops,self.arg_pos and
        return True, otherwise set self.funcprops to None and return False"""
        self.current_func = None
        if not self.config.arg_spec:
            return False
        func, arg_number = self._funcname_and_argnum(self.current_line)
        if not func:
            return False
        try:
            if inspection.is_eval_safe_name(func):
                f = self.get_object(func)
            else:
                try:
                    fake_cursor = self.current_line.index(func) + len(func)
                    f = simpleeval.evaluate_current_attribute(fake_cursor, self.current_line, self.interp.locals)
                except simpleeval.EvaluationError:
                    return False
            if inspect.isclass(f):
                class_f = None
                if (not class_f or not inspection.getfuncprops(func, class_f)) and hasattr(f, '__new__') and (f.__new__ is not object.__new__) and (f.__new__.__class__ is not object.__new__.__class__):
                    class_f = f.__new__
                if class_f:
                    f = class_f
        except Exception:
            return False
        self.current_func = f
        self.funcprops = inspection.getfuncprops(func, f)
        if self.funcprops:
            self.arg_pos = arg_number
            return True
        self.arg_pos = None
        return False

    def get_source_of_current_name(self) -> str:
        """Return the unicode source code of the object which is bound to the
        current name in the current input line. Throw `SourceNotFound` if the
        source cannot be found."""
        obj: Optional[Callable] = self.current_func
        try:
            if obj is None:
                line = self.current_line
                if not line.strip():
                    raise SourceNotFound(_('Nothing to get source of'))
                if inspection.is_eval_safe_name(line):
                    obj = self.get_object(line)
            return inspect.getsource(obj)
        except (AttributeError, NameError) as e:
            msg = _('Cannot get source: %s') % (e,)
        except OSError as e:
            msg = f'{e}'
        except TypeError as e:
            if 'built-in' in f'{e}':
                msg = _('Cannot access source of %r') % (obj,)
            else:
                msg = _('No source code found for %s') % (self.current_line,)
        raise SourceNotFound(msg)

    def set_docstring(self) -> None:
        self.docstring = None
        if not self.get_args():
            self.funcprops = None
        if self.current_func is not None:
            try:
                self.docstring = pydoc.getdoc(self.current_func)
            except IndexError:
                self.docstring = None
            else:
                if not self.docstring:
                    self.docstring = None

    def complete(self, tab: bool=False) -> Optional[bool]:
        """Construct a full list of possible completions and
        display them in a window. Also check if there's an available argspec
        (via the inspect module) and bang that on top of the completions too.
        The return value is whether the list_win is visible or not.

        If no matches are found, just return whether there's an argspec to show
        If any matches are found, save them and select the first one.

        If tab is True exactly one match found, make the replacement and return
          the result of running complete() again on the new line.
        """
        self.set_docstring()
        matches, completer = autocomplete.get_completer(self.completers, cursor_offset=self.cursor_offset, line=self.current_line, locals_=cast(Dict[str, Any], self.interp.locals), argspec=self.funcprops, current_block='\n'.join(self.buffer + [self.current_line]), complete_magic_methods=self.config.complete_magic_methods, history=self.history)
        if len(matches) == 0:
            self.matches_iter.clear()
            return bool(self.funcprops)
        if completer:
            self.matches_iter.update(self.cursor_offset, self.current_line, matches, completer)
            if len(matches) == 1:
                if tab:
                    self._cursor_offset, self._current_line = self.matches_iter.substitute_cseq()
                    return Repl.complete(self)
                elif self.matches_iter.current_word == matches[0]:
                    self.matches_iter.clear()
                    return False
                return completer.shown_before_tab
            else:
                return tab or completer.shown_before_tab
        else:
            return False

    def format_docstring(self, docstring: str, width: int, height: int) -> List[str]:
        """Take a string and try to format it into a sane list of strings to be
        put into the suggestion box."""
        lines = docstring.split('\n')
        out = []
        i = 0
        for line in lines:
            i += 1
            if not line.strip():
                out.append('\n')
            for block in textwrap.wrap(line, width):
                out.append('  ' + block + '\n')
                if i >= height:
                    return out
                i += 1
        out[-1] = out[-1].rstrip()
        return out

    def next_indentation(self) -> int:
        """Return the indentation of the next line based on the current
        input buffer."""
        if self.buffer:
            indentation = next_indentation(self.buffer[-1], self.config.tab_length)
            if indentation and self.config.dedent_after > 0:

                def line_is_empty(line):
                    return not line.strip()
                empty_lines = takewhile(line_is_empty, reversed(self.buffer))
                if sum((1 for _ in empty_lines)) >= self.config.dedent_after:
                    indentation -= 1
        else:
            indentation = 0
        return indentation

    @abstractmethod
    def getstdout(self) -> str:
        raise NotImplementedError()

    def get_session_formatted_for_file(self) -> str:
        """Format the stdout buffer to something suitable for writing to disk,
        i.e. without >>> and ... at input lines and with "# OUT: " prepended to
        output lines and "### " prepended to current line"""
        session_output = self.getstdout()

        def process():
            for line in session_output.split('\n'):
                if line.startswith(self.ps1):
                    yield line[len(self.ps1):]
                elif line.startswith(self.ps2):
                    yield line[len(self.ps2):]
                elif line.rstrip():
                    yield f'# OUT: {line}'
        return '\n'.join(process())

    def write2file(self) -> None:
        """Prompt for a filename and write the current contents of the stdout
        buffer to disk."""
        try:
            fn = self.interact.file_prompt(_('Save to file (Esc to cancel): '))
            if not fn:
                self.interact.notify(_('Save cancelled.'))
                return
        except ValueError:
            self.interact.notify(_('Save cancelled.'))
            return
        path = Path(fn).expanduser()
        if path.suffix != '.py' and self.config.save_append_py:
            path = Path(f'{path}.py')
        mode = 'w'
        if path.exists():
            new_mode = self.interact.file_prompt(_('%s already exists. Do you want to (c)ancel, (o)verwrite or (a)ppend? ') % (path,))
            if new_mode in ('o', 'overwrite', _('overwrite')):
                mode = 'w'
            elif new_mode in ('a', 'append', _('append')):
                mode = 'a'
            else:
                self.interact.notify(_('Save cancelled.'))
                return
        stdout_text = self.get_session_formatted_for_file()
        try:
            with open(path, mode) as f:
                f.write(stdout_text)
        except OSError as e:
            self.interact.notify(_("Error writing file '%s': %s") % (path, e))
        else:
            self.interact.notify(_('Saved to %s.') % (path,))

    def copy2clipboard(self) -> None:
        """Copy current content to clipboard."""
        if not have_pyperclip:
            self.interact.notify(_('No clipboard available.'))
            return
        content = self.get_session_formatted_for_file()
        try:
            pyperclip.copy(content)
        except pyperclip.PyperclipException:
            self.interact.notify(_('Could not copy to clipboard.'))
        else:
            self.interact.notify(_('Copied content to clipboard.'))

    def pastebin(self, s=None) -> Optional[str]:
        """Upload to a pastebin and display the URL in the status bar."""
        if s is None:
            s = self.getstdout()
        if self.config.pastebin_confirm and (not self.interact.confirm(_('Pastebin buffer? (y/N) '))):
            self.interact.notify(_('Pastebin aborted.'))
            return None
        else:
            return self.do_pastebin(s)

    def do_pastebin(self, s) -> Optional[str]:
        """Actually perform the upload."""
        paste_url: str
        if s == self.prev_pastebin_content:
            self.interact.notify(_('Duplicate pastebin. Previous URL: %s. Removal URL: %s') % (self.prev_pastebin_url, self.prev_removal_url), 10)
            return self.prev_pastebin_url
        self.interact.notify(_('Posting data to pastebin...'))
        try:
            paste_url, removal_url = self.paster.paste(s)
        except PasteFailed as e:
            self.interact.notify(_('Upload failed: %s') % e)
            return None
        self.prev_pastebin_content = s
        self.prev_pastebin_url = paste_url
        self.prev_removal_url = removal_url if removal_url is not None else ''
        if removal_url is not None:
            self.interact.notify(_('Pastebin URL: %s - Removal URL: %s') % (paste_url, removal_url), 10)
        else:
            self.interact.notify(_('Pastebin URL: %s') % (paste_url,), 10)
        return paste_url

    def push(self, s, insert_into_history=True) -> bool:
        """Push a line of code onto the buffer so it can process it all
        at once when a code block ends"""
        s = s.rstrip('\n')
        self.buffer.append(s)
        if insert_into_history:
            self.insert_into_history(s)
        more: bool = self.interp.runsource('\n'.join(self.buffer))
        if not more:
            self.buffer = []
        return more

    def insert_into_history(self, s: str):
        try:
            self.rl_history.append_reload_and_write(s, self.config.hist_file, getpreferredencoding())
        except RuntimeError as e:
            self.interact.notify(f'{e}')

    def prompt_undo(self) -> int:
        """Returns how many lines to undo, 0 means don't undo"""
        if self.config.single_undo_time < 0 or self.interp.timer.estimate() < self.config.single_undo_time:
            return 1
        est = self.interp.timer.estimate()
        m = self.interact.file_prompt(_('Undo how many lines? (Undo will take up to ~%.1f seconds) [1]') % (est,))
        if m is None:
            self.interact.notify(_('Undo canceled'), 0.1)
            return 0
        try:
            if m == '':
                m = '1'
            n = int(m)
        except ValueError:
            self.interact.notify(_('Undo canceled'), 0.1)
            return 0
        else:
            if n == 0:
                self.interact.notify(_('Undo canceled'), 0.1)
                return 0
            else:
                message = ngettext('Undoing %d line... (est. %.1f seconds)', 'Undoing %d lines... (est. %.1f seconds)', n)
                self.interact.notify(message % (n, est), 0.1)
            return n

    def undo(self, n: int=1) -> None:
        """Go back in the undo history n steps and call reevaluate()
        Note that in the program this is called "Rewind" because I
        want it to be clear that this is by no means a true undo
        implementation, it is merely a convenience bonus."""
        if not self.history:
            return None
        self.interp.timer.reset_timer()
        if len(self.history) < n:
            n = len(self.history)
        entries = list(self.rl_history.entries)
        last_entries = self.history[-n:]
        last_entries.reverse()
        self.redo_stack += last_entries
        self.history = self.history[:-n]
        self.reevaluate()
        self.rl_history.entries = entries

    def flush(self) -> None:
        """Olivier Grisel brought it to my attention that the logging
        module tries to call this method, since it makes assumptions
        about stdout that may not necessarily be true. The docs for
        sys.stdout say:

        "stdout and stderr needn't be built-in file objects: any
         object is acceptable as long as it has a write() method
         that takes a string argument."

        So I consider this to be a bug in logging, and this is a hack
        to fix it, unfortunately. I'm sure it's not the only module
        to do it."""

    def close(self):
        """See the flush() method docstring."""

    def tokenize(self, s, newline=False) -> List[Tuple[_TokenType, str]]:
        """Tokenizes a line of code, returning pygments tokens
        with side effects/impurities:
        - reads self.cpos to see what parens should be highlighted
        - reads self.buffer to see what came before the passed in line
        - sets self.highlighted_paren to (buffer_lineno, tokens_for_that_line)
          for buffer line that should replace that line to unhighlight it,
          or None if no paren is currently highlighted
        - calls reprint_line with a buffer's line's tokens and the buffer
          lineno that has changed if line other than the current line changes
        """
        highlighted_paren = None
        source = '\n'.join(self.buffer + [s])
        cursor = len(source) - self.cpos
        if self.cpos:
            cursor += 1
        stack: List[Any] = list()
        all_tokens = list(Python3Lexer().get_tokens(source))
        while not all_tokens[-1][1]:
            all_tokens.pop()
        all_tokens[-1] = (all_tokens[-1][0], all_tokens[-1][1].rstrip('\n'))
        line = pos = 0
        parens = dict(zip('{([', '})]'))
        line_tokens: List[Tuple[_TokenType, str]] = list()
        saved_tokens: List[Tuple[_TokenType, str]] = list()
        search_for_paren = True
        for token, value in split_lines(all_tokens):
            pos += len(value)
            if token is Token.Text and value == '\n':
                line += 1
                line_tokens = list()
                saved_tokens = list()
                continue
            line_tokens.append((token, value))
            saved_tokens.append((token, value))
            if not search_for_paren:
                continue
            under_cursor = pos == cursor
            if token is Token.Punctuation:
                if value in parens:
                    if under_cursor:
                        line_tokens[-1] = (Parenthesis.UnderCursor, value)
                        stack.append((Parenthesis, value))
                    else:
                        stack.append((line, len(line_tokens) - 1, line_tokens, value))
                elif value in parens.values():
                    saved_stack = list(stack)
                    try:
                        while True:
                            opening = stack.pop()
                            if parens[opening[-1]] == value:
                                break
                    except IndexError:
                        opening = None
                        if not saved_stack:
                            search_for_paren = False
                        else:
                            stack = saved_stack
                    if opening and opening[0] is Parenthesis:
                        line_tokens[-1] = (Parenthesis, value)
                        search_for_paren = False
                    elif opening and under_cursor and (not newline):
                        if self.cpos:
                            line_tokens[-1] = (Parenthesis.UnderCursor, value)
                        else:
                            line_tokens[-1] = (Parenthesis, value)
                        lineno, i, tokens, opening = opening
                        if lineno == len(self.buffer):
                            highlighted_paren = (lineno, saved_tokens)
                            line_tokens[i] = (Parenthesis, opening)
                        else:
                            highlighted_paren = (lineno, list(tokens))
                            tokens[i] = (Parenthesis, opening)
                            self.reprint_line(lineno, tokens)
                        search_for_paren = False
                elif under_cursor:
                    search_for_paren = False
        self.highlighted_paren = highlighted_paren
        if line != len(self.buffer):
            return list()
        return line_tokens

    def clear_current_line(self) -> None:
        """This is used as the exception callback for the Interpreter instance.
        It prevents autoindentation from occurring after a traceback."""

    def send_to_external_editor(self, text: str) -> str:
        """Returns modified text from an editor, or the original text if editor
        exited with non-zero"""
        encoding = getpreferredencoding()
        editor_args = shlex.split(self.config.editor)
        with tempfile.NamedTemporaryFile(suffix='.py') as temp:
            temp.write(text.encode(encoding))
            temp.flush()
            args = editor_args + [temp.name]
            if subprocess.call(args) == 0:
                with open(temp.name) as f:
                    return f.read()
            else:
                return text

    def open_in_external_editor(self, filename):
        editor_args = shlex.split(self.config.editor)
        args = editor_args + [filename]
        return subprocess.call(args) == 0

    def edit_config(self):
        if not self.config.config_path.is_file():
            if self.interact.confirm(_('Config file does not exist - create new from default? (y/N)')):
                try:
                    default_config = pkgutil.get_data('bpython', 'sample-config')
                    default_config = default_config.decode('ascii')
                    containing_dir = self.config.config_path.parent
                    if not containing_dir.exists():
                        containing_dir.mkdir(parents=True)
                    with open(self.config.config_path, 'w') as f:
                        f.write(default_config)
                except OSError as e:
                    self.interact.notify(_("Error writing file '%s': %s") % (self.config.config_path, e))
                    return False
            else:
                return False
        try:
            if self.open_in_external_editor(self.config.config_path):
                self.interact.notify(_('bpython config file edited. Restart bpython for changes to take effect.'))
        except OSError as e:
            self.interact.notify(_('Error editing config file: %s') % e)