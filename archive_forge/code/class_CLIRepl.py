import curses
import errno
import functools
import math
import os
import platform
import re
import struct
import sys
import time
from typing import (
from ._typing_compat import Literal
import unicodedata
from dataclasses import dataclass
from pygments import format
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from .formatter import BPythonFormatter
from .config import getpreferredencoding, Config
from .keys import cli_key_dispatch as key_dispatch
from . import translations
from .translations import _
from . import repl, inspection
from . import args as bpargs
from .pager import page
from .args import parse as argsparse
class CLIRepl(repl.Repl):

    def __init__(self, scr: '_CursesWindow', interp: repl.Interpreter, statusbar: 'Statusbar', config: Config, idle: Optional[Callable]=None):
        super().__init__(interp, config)
        self.interp.writetb = self.writetb
        self.scr: '_CursesWindow' = scr
        self.stdout_hist = ''
        self.list_win = newwin(get_colpair(config, 'background'), 1, 1, 1, 1)
        self.cpos = 0
        self.do_exit = False
        self.exit_value: Tuple[Any, ...] = ()
        self.f_string = ''
        self.idle = idle
        self.in_hist = False
        self.paste_mode = False
        self.last_key_press = time.time()
        self.s = ''
        self.statusbar = statusbar
        self.formatter = BPythonFormatter(config.color_scheme)
        self.interact = CLIInteraction(self.config, statusbar=self.statusbar)
        self.ix: int
        self.iy: int
        self.arg_pos: Union[str, int, None]
        self.prev_block_finished: int
        if config.cli_suggestion_width <= 0 or config.cli_suggestion_width > 1:
            config.cli_suggestion_width = 0.8

    def _get_cursor_offset(self) -> int:
        return len(self.s) - self.cpos

    def _set_cursor_offset(self, offset: int) -> None:
        self.cpos = len(self.s) - offset

    def addstr(self, s: str) -> None:
        """Add a string to the current input line and figure out
        where it should go, depending on the cursor position."""
        self.rl_history.reset()
        if not self.cpos:
            self.s += s
        else:
            l = len(self.s)
            self.s = self.s[:l - self.cpos] + s + self.s[l - self.cpos:]
        self.complete()

    def atbol(self) -> bool:
        """Return True or False accordingly if the cursor is at the beginning
        of the line (whitespace is ignored). This exists so that p_key() knows
        how to handle the tab key being pressed - if there is nothing but white
        space before the cursor then process it as a normal tab otherwise
        attempt tab completion."""
        return not self.s.lstrip()

    def bs(self, delete_tabs: bool=True) -> int:
        """Process a backspace"""
        self.rl_history.reset()
        y, x = self.scr.getyx()
        if not self.s:
            return None
        if x == self.ix and y == self.iy:
            return None
        n = 1
        self.clear_wrapped_lines()
        if not self.cpos:
            if self.atbol() and delete_tabs:
                n = len(self.s) % self.config.tab_length
                if not n:
                    n = self.config.tab_length
            self.s = self.s[:-n]
        else:
            self.s = self.s[:-self.cpos - 1] + self.s[-self.cpos:]
        self.print_line(self.s, clr=True)
        return n

    def bs_word(self) -> str:
        self.rl_history.reset()
        pos = len(self.s) - self.cpos - 1
        deleted = []
        while pos >= 0 and self.s[pos] == ' ':
            deleted.append(self.s[pos])
            pos -= self.bs()
        while pos >= 0 and self.s[pos] != ' ':
            deleted.append(self.s[pos])
            pos -= self.bs()
        return ''.join(reversed(deleted))

    def check(self) -> None:
        """Check if paste mode should still be active and, if not, deactivate
        it and force syntax highlighting."""
        if self.paste_mode and time.time() - self.last_key_press > self.config.paste_time:
            self.paste_mode = False
            self.print_line(self.s)

    def clear_current_line(self) -> None:
        """Called when a SyntaxError occurred in the interpreter. It is
        used to prevent autoindentation from occurring after a
        traceback."""
        repl.Repl.clear_current_line(self)
        self.s = ''

    def clear_wrapped_lines(self) -> None:
        """Clear the wrapped lines of the current input."""
        height, width = self.scr.getmaxyx()
        max_y = min(self.iy + (self.ix + len(self.s)) // width + 1, height)
        for y in range(self.iy + 1, max_y):
            self.scr.move(y, 0)
            self.scr.clrtoeol()

    def complete(self, tab: bool=False) -> None:
        """Get Autocomplete list and window.

        Called whenever these should be updated, and called
        with tab
        """
        if self.paste_mode:
            self.scr.touchwin()
            return
        list_win_visible = repl.Repl.complete(self, tab)
        if list_win_visible:
            try:
                f = None
                if self.matches_iter.completer:
                    f = self.matches_iter.completer.format
                self.show_list(self.matches_iter.matches, self.arg_pos, topline=self.funcprops, formatter=f)
            except curses.error:
                self.list_win.border()
                self.list_win.refresh()
                list_win_visible = False
        if not list_win_visible:
            self.scr.redrawwin()
            self.scr.refresh()

    def clrtobol(self) -> None:
        """Clear from cursor to beginning of line; usual C-u behaviour"""
        self.clear_wrapped_lines()
        if not self.cpos:
            self.s = ''
        else:
            self.s = self.s[-self.cpos:]
        self.print_line(self.s, clr=True)
        self.scr.redrawwin()
        self.scr.refresh()

    def _get_current_line(self) -> str:
        return self.s

    def _set_current_line(self, line: str) -> None:
        self.s = line

    def cut_to_buffer(self) -> None:
        """Clear from cursor to end of line, placing into cut buffer"""
        self.cut_buffer = self.s[-self.cpos:]
        self.s = self.s[:-self.cpos]
        self.cpos = 0
        self.print_line(self.s, clr=True)
        self.scr.redrawwin()
        self.scr.refresh()

    def delete(self) -> None:
        """Process a del"""
        if not self.s:
            return
        if self.mvc(-1):
            self.bs(False)

    def echo(self, s: str, redraw: bool=True) -> None:
        """Parse and echo a formatted string with appropriate attributes. It
        uses the formatting method as defined in formatter.py to parse the
        strings. It won't update the screen if it's reevaluating the code (as it
        does with undo)."""
        a = get_colpair(self.config, 'output')
        if '\x01' in s:
            rx = re.search('\x01([A-Za-z])([A-Za-z]?)', s)
            if rx:
                fg = rx.groups()[0]
                bg = rx.groups()[1]
                col_num = self._C[fg.lower()]
                if bg and bg != 'I':
                    col_num *= self._C[bg.lower()]
                a = curses.color_pair(int(col_num) + 1)
                if bg == 'I':
                    a = a | curses.A_REVERSE
                s = re.sub('\x01[A-Za-z][A-Za-z]?', '', s)
                if fg.isupper():
                    a = a | curses.A_BOLD
        s = s.replace('\x03', '')
        s = s.replace('\x01', '')
        s = s.replace('\x00', '')
        s = s.replace('\r\n', '\n')
        self.scr.addstr(s, a)
        if redraw and (not self.evaluating):
            self.scr.refresh()

    def end(self, refresh: bool=True) -> bool:
        self.cpos = 0
        h, w = gethw()
        y, x = divmod(len(self.s) + self.ix, w)
        y += self.iy
        self.scr.move(y, x)
        if refresh:
            self.scr.refresh()
        return True

    def hbegin(self) -> None:
        """Replace the active line with first line in history and
        increment the index to keep track"""
        self.cpos = 0
        self.clear_wrapped_lines()
        self.rl_history.enter(self.s)
        self.s = self.rl_history.first()
        self.print_line(self.s, clr=True)

    def hend(self) -> None:
        """Same as hbegin() but, well, forward"""
        self.cpos = 0
        self.clear_wrapped_lines()
        self.rl_history.enter(self.s)
        self.s = self.rl_history.last()
        self.print_line(self.s, clr=True)

    def back(self) -> None:
        """Replace the active line with previous line in history and
        increment the index to keep track"""
        self.cpos = 0
        self.clear_wrapped_lines()
        self.rl_history.enter(self.s)
        self.s = self.rl_history.back()
        self.print_line(self.s, clr=True)

    def fwd(self) -> None:
        """Same as back() but, well, forward"""
        self.cpos = 0
        self.clear_wrapped_lines()
        self.rl_history.enter(self.s)
        self.s = self.rl_history.forward()
        self.print_line(self.s, clr=True)

    def search(self) -> None:
        """Search with the partial matches from the history object."""
        self.cpo = 0
        self.clear_wrapped_lines()
        self.rl_history.enter(self.s)
        self.s = self.rl_history.back(start=False, search=True)
        self.print_line(self.s, clr=True)

    def get_key(self) -> str:
        key = ''
        while True:
            try:
                key += self.scr.getkey()
                key = key.encode('latin-1').decode(getpreferredencoding())
                self.scr.nodelay(False)
            except UnicodeDecodeError:
                self.scr.nodelay(True)
            except curses.error:
                self.scr.nodelay(False)
                if key:
                    return key
            else:
                if key != '\x00':
                    t = time.time()
                    self.paste_mode = t - self.last_key_press <= self.config.paste_time
                    self.last_key_press = t
                    return key
                else:
                    key = ''
            finally:
                if self.idle:
                    self.idle(self)

    def get_line(self) -> str:
        """Get a line of text and return it
        This function initialises an empty string and gets the
        curses cursor position on the screen and stores it
        for the echo() function to use later (I think).
        Then it waits for key presses and passes them to p_key(),
        which returns None if Enter is pressed (that means "Return",
        idiot)."""
        self.s = ''
        self.rl_history.reset()
        self.iy, self.ix = self.scr.getyx()
        if not self.paste_mode:
            for _ in range(self.next_indentation()):
                self.p_key('\t')
        self.cpos = 0
        while True:
            key = self.get_key()
            if self.p_key(key) is None:
                if self.config.cli_trim_prompts and self.s.startswith('>>> '):
                    self.s = self.s[4:]
                return self.s

    def home(self, refresh: bool=True) -> bool:
        self.scr.move(self.iy, self.ix)
        self.cpos = len(self.s)
        if refresh:
            self.scr.refresh()
        return True

    def lf(self) -> None:
        """Process a linefeed character; it only needs to check the
        cursor position and move appropriately so it doesn't clear
        the current line after the cursor."""
        if self.cpos:
            for _ in range(self.cpos):
                self.mvc(-1)
        self.print_line(self.s, newline=True)
        self.echo('\n')

    def mkargspec(self, topline: inspection.FuncProps, in_arg: Union[str, int, None], down: bool) -> int:
        """This figures out what to do with the argspec and puts it nicely into
        the list window. It returns the number of lines used to display the
        argspec.  It's also kind of messy due to it having to call so many
        addstr() to get the colouring right, but it seems to be pretty
        sturdy."""
        r = 3
        fn = topline.func
        args = topline.argspec.args
        kwargs = topline.argspec.defaults
        _args = topline.argspec.varargs
        _kwargs = topline.argspec.varkwargs
        is_bound_method = topline.is_bound_method
        kwonly = topline.argspec.kwonly
        kwonly_defaults = topline.argspec.kwonly_defaults or dict()
        max_w = int(self.scr.getmaxyx()[1] * 0.6)
        self.list_win.erase()
        self.list_win.resize(3, max_w)
        h, w = self.list_win.getmaxyx()
        self.list_win.addstr('\n  ')
        self.list_win.addstr(fn, get_colpair(self.config, 'name') | curses.A_BOLD)
        self.list_win.addstr(': (', get_colpair(self.config, 'name'))
        maxh = self.scr.getmaxyx()[0]
        if is_bound_method and isinstance(in_arg, int):
            in_arg += 1
        punctuation_colpair = get_colpair(self.config, 'punctuation')
        for k, i in enumerate(args):
            y, x = self.list_win.getyx()
            ln = len(str(i))
            kw = None
            if kwargs and k + 1 > len(args) - len(kwargs):
                kw = repr(kwargs[k - (len(args) - len(kwargs))])
                ln += len(kw) + 1
            if ln + x >= w:
                ty = self.list_win.getbegyx()[0]
                if not down and ty > 0:
                    h += 1
                    self.list_win.mvwin(ty - 1, 1)
                    self.list_win.resize(h, w)
                elif down and h + r < maxh - ty:
                    h += 1
                    self.list_win.resize(h, w)
                else:
                    break
                r += 1
                self.list_win.addstr('\n\t')
            if str(i) == 'self' and k == 0:
                color = get_colpair(self.config, 'name')
            else:
                color = get_colpair(self.config, 'token')
            if k == in_arg or i == in_arg:
                color |= curses.A_BOLD
            self.list_win.addstr(str(i), color)
            if kw is not None:
                self.list_win.addstr('=', punctuation_colpair)
                self.list_win.addstr(kw, get_colpair(self.config, 'token'))
            if k != len(args) - 1:
                self.list_win.addstr(', ', punctuation_colpair)
        if _args:
            if args:
                self.list_win.addstr(', ', punctuation_colpair)
            self.list_win.addstr(f'*{_args}', get_colpair(self.config, 'token'))
        if kwonly:
            if not _args:
                if args:
                    self.list_win.addstr(', ', punctuation_colpair)
                self.list_win.addstr('*', punctuation_colpair)
            marker = object()
            for arg in kwonly:
                self.list_win.addstr(', ', punctuation_colpair)
                color = get_colpair(self.config, 'token')
                if arg == in_arg:
                    color |= curses.A_BOLD
                self.list_win.addstr(arg, color)
                default = kwonly_defaults.get(arg, marker)
                if default is not marker:
                    self.list_win.addstr('=', punctuation_colpair)
                    self.list_win.addstr(repr(default), get_colpair(self.config, 'token'))
        if _kwargs:
            if args or _args or kwonly:
                self.list_win.addstr(', ', punctuation_colpair)
            self.list_win.addstr(f'**{_kwargs}', get_colpair(self.config, 'token'))
        self.list_win.addstr(')', punctuation_colpair)
        return r

    def mvc(self, i: int, refresh: bool=True) -> bool:
        """This method moves the cursor relatively from the current
        position, where:
            0 == (right) end of current line
            length of current line len(self.s) == beginning of current line
        and:
            current cursor position + i
            for positive values of i the cursor will move towards the beginning
            of the line, negative values the opposite."""
        y, x = self.scr.getyx()
        if self.cpos == 0 and i < 0:
            return False
        if x == self.ix and y == self.iy and (i >= 1):
            return False
        h, w = gethw()
        if x - i < 0:
            y -= 1
            x = w
        if x - i >= w:
            y += 1
            x = 0 + i
        self.cpos += i
        self.scr.move(y, x - i)
        if refresh:
            self.scr.refresh()
        return True

    def p_key(self, key: str) -> Union[None, str, bool]:
        """Process a keypress"""
        if key is None:
            return ''
        config = self.config
        if platform.system() == 'Windows':
            C_BACK = chr(127)
            BACKSP = chr(8)
        else:
            C_BACK = chr(8)
            BACKSP = chr(127)
        if key == C_BACK:
            self.clrtobol()
            key = '\n'
        if key == chr(27):
            return ''
        if key in (BACKSP, 'KEY_BACKSPACE'):
            self.bs()
            self.complete()
            return ''
        elif key in key_dispatch[config.delete_key] and (not self.s):
            self.do_exit = True
            return None
        elif key in ('KEY_DC',) + key_dispatch[config.delete_key]:
            self.delete()
            self.complete()
            self.print_line(self.s)
            return ''
        elif key in key_dispatch[config.undo_key]:
            n = self.prompt_undo()
            if n > 0:
                self.undo(n=n)
            return ''
        elif key in key_dispatch[config.search_key]:
            self.search()
            return ''
        elif key in ('KEY_UP',) + key_dispatch[config.up_one_line_key]:
            self.back()
            return ''
        elif key in ('KEY_DOWN',) + key_dispatch[config.down_one_line_key]:
            self.fwd()
            return ''
        elif key in ('KEY_LEFT', ' ^B', chr(2)):
            self.mvc(1)
            self.print_line(self.s)
        elif key in ('KEY_RIGHT', '^F', chr(6)):
            self.mvc(-1)
            self.print_line(self.s)
        elif key in ('KEY_HOME', '^A', chr(1)):
            self.home()
            self.print_line(self.s)
        elif key in ('KEY_END', '^E', chr(5)):
            self.end()
            self.print_line(self.s)
        elif key in ('KEY_NPAGE',):
            self.hend()
            self.print_line(self.s)
        elif key in ('KEY_PPAGE',):
            self.hbegin()
            self.print_line(self.s)
        elif key in key_dispatch[config.cut_to_buffer_key]:
            self.cut_to_buffer()
            return ''
        elif key in key_dispatch[config.yank_from_buffer_key]:
            self.yank_from_buffer()
            return ''
        elif key in key_dispatch[config.clear_word_key]:
            self.cut_buffer = self.bs_word()
            self.complete()
            return ''
        elif key in key_dispatch[config.clear_line_key]:
            self.clrtobol()
            return ''
        elif key in key_dispatch[config.clear_screen_key]:
            self.screen_hist: List = [self.screen_hist[-1]]
            self.highlighted_paren = None
            self.redraw()
            return ''
        elif key in key_dispatch[config.exit_key]:
            if not self.s:
                self.do_exit = True
                return None
            else:
                return ''
        elif key in key_dispatch[config.save_key]:
            self.write2file()
            return ''
        elif key in key_dispatch[config.pastebin_key]:
            self.pastebin()
            return ''
        elif key in key_dispatch[config.copy_clipboard_key]:
            self.copy2clipboard()
            return ''
        elif key in key_dispatch[config.last_output_key]:
            page(self.stdout_hist[self.prev_block_finished:-4])
            return ''
        elif key in key_dispatch[config.show_source_key]:
            try:
                source = self.get_source_of_current_name()
            except repl.SourceNotFound as e:
                self.statusbar.message(f'{e}')
            else:
                if config.highlight_show_source:
                    source = format(Python3Lexer().get_tokens(source), TerminalFormatter())
                page(source)
            return ''
        elif key in ('\n', '\r', 'PADENTER'):
            self.lf()
            return None
        elif key == '\t':
            return self.tab()
        elif key == 'KEY_BTAB':
            return self.tab(back=True)
        elif key in key_dispatch[config.suspend_key]:
            if platform.system() != 'Windows':
                self.suspend()
                return ''
            else:
                self.do_exit = True
                return None
        elif key == '\x18':
            return self.send_current_line_to_editor()
        elif key == '\x03':
            raise KeyboardInterrupt()
        elif key[0:3] == 'PAD' and key not in ('PAD0', 'PADSTOP'):
            pad_keys = {'PADMINUS': '-', 'PADPLUS': '+', 'PADSLASH': '/', 'PADSTAR': '*'}
            try:
                self.addstr(pad_keys[key])
                self.print_line(self.s)
            except KeyError:
                return ''
        elif len(key) == 1 and (not unicodedata.category(key) == 'Cc'):
            self.addstr(key)
            self.print_line(self.s)
        else:
            return ''
        return True

    def print_line(self, s: Optional[str], clr: bool=False, newline: bool=False) -> None:
        """Chuck a line of text through the highlighter, move the cursor
        to the beginning of the line and output it to the screen."""
        if not s:
            clr = True
        if self.highlighted_paren is not None:
            lineno = self.highlighted_paren[0]
            tokens = self.highlighted_paren[1]
            self.reprint_line(lineno, tokens)
            self.highlighted_paren = None
        if self.config.syntax and (not self.paste_mode or newline):
            o = format(self.tokenize(s, newline), self.formatter)
        else:
            o = s
        self.f_string = o
        self.scr.move(self.iy, self.ix)
        if clr:
            self.scr.clrtoeol()
        if clr and (not s):
            self.scr.refresh()
        if o:
            for t in o.split('\x04'):
                self.echo(t.rstrip('\n'))
        if self.cpos:
            t = self.cpos
            for _ in range(self.cpos):
                self.mvc(1)
            self.cpos = t

    def prompt(self, more: Any) -> None:
        """Show the appropriate Python prompt"""
        if not more:
            self.echo('\x01{}\x03{}'.format(self.config.color_scheme['prompt'], self.ps1))
            self.stdout_hist += self.ps1
            self.screen_hist.append('\x01%s\x03%s\x04' % (self.config.color_scheme['prompt'], self.ps1))
        else:
            prompt_more_color = self.config.color_scheme['prompt_more']
            self.echo(f'\x01{prompt_more_color}\x03{self.ps2}')
            self.stdout_hist += self.ps2
            self.screen_hist.append(f'\x01{prompt_more_color}\x03{self.ps2}\x04')

    def push(self, s: str, insert_into_history: bool=True) -> bool:
        curses.raw(False)
        try:
            return super().push(s, insert_into_history)
        except SystemExit as e:
            self.do_exit = True
            self.exit_value = e.args
            return False
        finally:
            curses.raw(True)

    def redraw(self) -> None:
        """Redraw the screen using screen_hist"""
        self.scr.erase()
        for k, s in enumerate(self.screen_hist):
            if not s:
                continue
            self.iy, self.ix = self.scr.getyx()
            for i in s.split('\x04'):
                self.echo(i, redraw=False)
            if k < len(self.screen_hist) - 1:
                self.scr.addstr('\n')
        self.iy, self.ix = self.scr.getyx()
        self.print_line(self.s)
        self.scr.refresh()
        self.statusbar.refresh()

    def repl(self) -> Tuple[Any, ...]:
        """Initialise the repl and jump into the loop. This method also has to
        keep a stack of lines entered for the horrible "undo" feature. It also
        tracks everything that would normally go to stdout in the normal Python
        interpreter so it can quickly write it to stdout on exit after
        curses.endwin(), as well as a history of lines entered for using
        up/down to go back and forth (which has to be separate to the
        evaluation history, which will be truncated when undoing."""
        self.push('from bpython._internal import _help as help\n', False)
        self.iy, self.ix = self.scr.getyx()
        self.more = False
        while not self.do_exit:
            self.f_string = ''
            self.prompt(self.more)
            try:
                inp = self.get_line()
            except KeyboardInterrupt:
                self.statusbar.message('KeyboardInterrupt')
                self.scr.addstr('\n')
                self.scr.touchwin()
                self.scr.refresh()
                continue
            self.scr.redrawwin()
            if self.do_exit:
                return self.exit_value
            self.history.append(inp)
            self.screen_hist[-1] += self.f_string
            self.stdout_hist += inp + '\n'
            stdout_position = len(self.stdout_hist)
            self.more = self.push(inp)
            if not self.more:
                self.prev_block_finished = stdout_position
                self.s = ''
        return self.exit_value

    def reprint_line(self, lineno: int, tokens: List[Tuple[_TokenType, str]]) -> None:
        """Helper function for paren highlighting: Reprint line at offset
        `lineno` in current input buffer."""
        if not self.buffer or lineno == len(self.buffer):
            return
        real_lineno = self.iy
        height, width = self.scr.getmaxyx()
        for i in range(lineno, len(self.buffer)):
            string = self.buffer[i]
            length = len(string.encode(getpreferredencoding())) + 4
            real_lineno -= int(math.ceil(length / width))
        if real_lineno < 0:
            return
        self.scr.move(real_lineno, len(self.ps1) if lineno == 0 else len(self.ps2))
        line = format(tokens, BPythonFormatter(self.config.color_scheme))
        for string in line.split('\x04'):
            self.echo(string)

    def resize(self) -> None:
        """This method exists simply to keep it straight forward when
        initialising a window and resizing it."""
        self.size()
        self.scr.erase()
        self.scr.resize(self.h, self.w)
        self.scr.mvwin(self.y, self.x)
        self.statusbar.resize(refresh=False)
        self.redraw()

    def getstdout(self) -> str:
        """This method returns the 'spoofed' stdout buffer, for writing to a
        file or sending to a pastebin or whatever."""
        return self.stdout_hist + '\n'

    def reevaluate(self) -> None:
        """Clear the buffer, redraw the screen and re-evaluate the history"""
        self.evaluating = True
        self.stdout_hist = ''
        self.f_string = ''
        self.buffer: List[str] = []
        self.scr.erase()
        self.screen_hist = []
        self.cpos = -1
        self.prompt(False)
        self.iy, self.ix = self.scr.getyx()
        for line in self.history:
            self.stdout_hist += line + '\n'
            self.print_line(line)
            self.screen_hist[-1] += self.f_string
            self.scr.addstr('\n')
            self.more = self.push(line)
            self.prompt(self.more)
            self.iy, self.ix = self.scr.getyx()
        self.cpos = 0
        indent = repl.next_indentation(self.s, self.config.tab_length)
        self.s = ''
        self.scr.refresh()
        if self.buffer:
            for _ in range(indent):
                self.tab()
        self.evaluating = False

    def write(self, s: str) -> None:
        """For overriding stdout defaults"""
        if '\x04' in s:
            for block in s.split('\x04'):
                self.write(block)
            return
        if s.rstrip() and '\x03' in s:
            t = s.split('\x03')[1]
        else:
            t = s
        if not self.stdout_hist:
            self.stdout_hist = t
        else:
            self.stdout_hist += t
        self.echo(s)
        self.screen_hist.append(s.rstrip())

    def show_list(self, items: List[str], arg_pos: Union[str, int, None], topline: Optional[inspection.FuncProps]=None, formatter: Optional[Callable]=None, current_item: Optional[str]=None) -> None:
        v_items: Collection
        shared = ShowListState()
        y, x = self.scr.getyx()
        h, w = self.scr.getmaxyx()
        down = y < h // 2
        if down:
            max_h = h - y
        else:
            max_h = y + 1
        max_w = int(w * self.config.cli_suggestion_width)
        self.list_win.erase()
        if items and formatter:
            items = [formatter(x) for x in items]
            if current_item is not None:
                current_item = formatter(current_item)
        if topline:
            height_offset = self.mkargspec(topline, arg_pos, down) + 1
        else:
            height_offset = 0

        def lsize() -> bool:
            wl = max((len(i) for i in v_items)) + 1
            if not wl:
                wl = 1
            cols = (max_w - 2) // wl or 1
            rows = len(v_items) // cols
            if cols * rows < len(v_items):
                rows += 1
            if rows + 2 >= max_h:
                return False
            shared.rows = rows
            shared.cols = cols
            shared.wl = wl
            return True
        if items:
            v_items = [items[0][:max_w - 3]]
            lsize()
        else:
            v_items = []
        for i in items[1:]:
            v_items.append(i[:max_w - 3])
            if not lsize():
                del v_items[-1]
                v_items[-1] = '...'
                break
        rows = shared.rows
        if rows + height_offset < max_h:
            rows += height_offset
            display_rows = rows
        else:
            display_rows = rows + height_offset
        cols = shared.cols
        wl = shared.wl
        if topline and (not v_items):
            w = max_w
        elif wl + 3 > max_w:
            w = max_w
        else:
            t = (cols + 1) * wl + 3
            if t > max_w:
                t = max_w
            w = t
        if height_offset and display_rows + 5 >= max_h:
            del v_items[-(cols * height_offset):]
        if self.docstring is None:
            self.list_win.resize(rows + 2, w)
        else:
            docstring = self.format_docstring(self.docstring, max_w - 2, max_h - height_offset)
            docstring_string = ''.join(docstring)
            rows += len(docstring)
            self.list_win.resize(rows, max_w)
        if down:
            self.list_win.mvwin(y + 1, 0)
        else:
            self.list_win.mvwin(y - rows - 2, 0)
        if v_items:
            self.list_win.addstr('\n ')
        for ix, i in enumerate(v_items):
            padding = (wl - len(i)) * ' '
            if i == current_item:
                color = get_colpair(self.config, 'operator')
            else:
                color = get_colpair(self.config, 'main')
            self.list_win.addstr(i + padding, color)
            if (cols == 1 or (ix and (not (ix + 1) % cols))) and ix + 1 < len(v_items):
                self.list_win.addstr('\n ')
        if self.docstring is not None:
            self.list_win.addstr('\n' + docstring_string, get_colpair(self.config, 'comment'))
        y = self.list_win.getyx()[0]
        self.list_win.resize(y + 2, w)
        self.statusbar.win.touchwin()
        self.statusbar.win.noutrefresh()
        self.list_win.attron(get_colpair(self.config, 'main'))
        self.list_win.border()
        self.scr.touchwin()
        self.scr.cursyncup()
        self.scr.noutrefresh()
        self.scr.move(*self.scr.getyx())
        self.list_win.refresh()

    def size(self) -> None:
        """Set instance attributes for x and y top left corner coordinates
        and width and height for the window."""
        global stdscr
        if stdscr:
            h, w = stdscr.getmaxyx()
        self.y: int = 0
        self.w: int = w
        self.h: int = h - 1
        self.x: int = 0

    def suspend(self) -> None:
        """Suspend the current process for shell job control."""
        if platform.system() != 'Windows':
            curses.endwin()
            os.kill(os.getpid(), signal.SIGSTOP)

    def tab(self, back: bool=False) -> bool:
        """Process the tab key being hit.

        If there's only whitespace
        in the line or the line is blank then process a normal tab,
        otherwise attempt to autocomplete to the best match of possible
        choices in the match list.

        If `back` is True, walk backwards through the list of suggestions
        and don't indent if there are only whitespace in the line.
        """
        if self.atbol() and (not back):
            x_pos = len(self.s) - self.cpos
            num_spaces = x_pos % self.config.tab_length
            if not num_spaces:
                num_spaces = self.config.tab_length
            self.addstr(' ' * num_spaces)
            self.print_line(self.s)
            return True
        if not self.matches_iter:
            self.complete(tab=True)
            self.print_line(self.s)
        if self.matches_iter.is_cseq():
            temp_cursor_offset, self.s = self.matches_iter.substitute_cseq()
            self.cursor_offset = temp_cursor_offset
            self.print_line(self.s)
            if not self.matches_iter:
                self.complete()
        elif self.matches_iter.matches:
            current_match = self.matches_iter.previous() if back else next(self.matches_iter)
            try:
                f = None
                if self.matches_iter.completer:
                    f = self.matches_iter.completer.format
                self.show_list(self.matches_iter.matches, self.arg_pos, topline=self.funcprops, formatter=f, current_item=current_match)
            except curses.error:
                self.list_win.border()
                self.list_win.refresh()
            _, self.s = self.matches_iter.cur_line()
            self.print_line(self.s, True)
        return True

    def undo(self, n: int=1) -> None:
        repl.Repl.undo(self, n)
        self.print_line(self.s)

    def writetb(self, lines: List[str]) -> None:
        for line in lines:
            self.write('\x01{}\x03{}'.format(self.config.color_scheme['error'], line))

    def yank_from_buffer(self) -> None:
        """Paste the text from the cut buffer at the current cursor location"""
        self.addstr(self.cut_buffer)
        self.print_line(self.s, clr=True)

    def send_current_line_to_editor(self) -> str:
        lines = self.send_to_external_editor(self.s).split('\n')
        self.s = ''
        self.print_line(self.s)
        while lines and (not lines[-1]):
            lines.pop()
        if not lines:
            return ''
        self.f_string = ''
        self.cpos = -1
        self.iy, self.ix = self.scr.getyx()
        self.evaluating = True
        for line in lines:
            self.stdout_hist += line + '\n'
            self.history.append(line)
            self.print_line(line)
            self.screen_hist[-1] += self.f_string
            self.scr.addstr('\n')
            self.more = self.push(line)
            self.prompt(self.more)
            self.iy, self.ix = self.scr.getyx()
        self.evaluating = False
        self.cpos = 0
        indent = repl.next_indentation(self.s, self.config.tab_length)
        self.s = ''
        self.scr.refresh()
        if self.buffer:
            for _ in range(indent):
                self.tab()
        self.print_line(self.s)
        self.scr.redrawwin()
        return ''