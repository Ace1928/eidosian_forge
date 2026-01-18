from __future__ import unicode_literals
from prompt_toolkit.filters import to_simple_filter, Condition
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from six.moves import range
import array
import errno
import os
import six
class Vt100_Output(Output):
    """
    :param get_size: A callable which returns the `Size` of the output terminal.
    :param stdout: Any object with has a `write` and `flush` method + an 'encoding' property.
    :param true_color: Use 24bit color instead of 256 colors. (Can be a :class:`SimpleFilter`.)
        When `ansi_colors_only` is set, only 16 colors are used.
    :param ansi_colors_only: Restrict to 16 ANSI colors only.
    :param term: The terminal environment variable. (xterm, xterm-256color, linux, ...)
    :param write_binary: Encode the output before writing it. If `True` (the
        default), the `stdout` object is supposed to expose an `encoding` attribute.
    """

    def __init__(self, stdout, get_size, true_color=False, ansi_colors_only=None, term=None, write_binary=True):
        assert callable(get_size)
        assert term is None or isinstance(term, six.text_type)
        assert all((hasattr(stdout, a) for a in ('write', 'flush')))
        if write_binary:
            assert hasattr(stdout, 'encoding')
        self._buffer = []
        self.stdout = stdout
        self.write_binary = write_binary
        self.get_size = get_size
        self.true_color = to_simple_filter(true_color)
        self.term = term or 'xterm'
        if ansi_colors_only is None:
            ANSI_COLORS_ONLY = bool(os.environ.get('PROMPT_TOOLKIT_ANSI_COLORS_ONLY', False))

            @Condition
            def ansi_colors_only():
                return ANSI_COLORS_ONLY or term in ('linux', 'eterm-color')
        else:
            ansi_colors_only = to_simple_filter(ansi_colors_only)
        self.ansi_colors_only = ansi_colors_only
        self._escape_code_cache = _EscapeCodeCache(ansi_colors_only=ansi_colors_only)
        self._escape_code_cache_true_color = _EscapeCodeCache(true_color=True, ansi_colors_only=ansi_colors_only)

    @classmethod
    def from_pty(cls, stdout, true_color=False, ansi_colors_only=None, term=None):
        """
        Create an Output class from a pseudo terminal.
        (This will take the dimensions by reading the pseudo
        terminal attributes.)
        """
        assert stdout.isatty()

        def get_size():
            rows, columns = _get_size(stdout.fileno())
            return Size(rows=rows, columns=columns)
        return cls(stdout, get_size, true_color=true_color, ansi_colors_only=ansi_colors_only, term=term)

    def fileno(self):
        """ Return file descriptor. """
        return self.stdout.fileno()

    def encoding(self):
        """ Return encoding used for stdout. """
        return self.stdout.encoding

    def write_raw(self, data):
        """
        Write raw data to output.
        """
        self._buffer.append(data)

    def write(self, data):
        """
        Write text to output.
        (Removes vt100 escape codes. -- used for safely writing text.)
        """
        self._buffer.append(data.replace('\x1b', '?'))

    def set_title(self, title):
        """
        Set terminal title.
        """
        if self.term not in ('linux', 'eterm-color'):
            self.write_raw('\x1b]2;%s\x07' % title.replace('\x1b', '').replace('\x07', ''))

    def clear_title(self):
        self.set_title('')

    def erase_screen(self):
        """
        Erases the screen with the background colour and moves the cursor to
        home.
        """
        self.write_raw('\x1b[2J')

    def enter_alternate_screen(self):
        self.write_raw('\x1b[?1049h\x1b[H')

    def quit_alternate_screen(self):
        self.write_raw('\x1b[?1049l')

    def enable_mouse_support(self):
        self.write_raw('\x1b[?1000h')
        self.write_raw('\x1b[?1015h')
        self.write_raw('\x1b[?1006h')

    def disable_mouse_support(self):
        self.write_raw('\x1b[?1000l')
        self.write_raw('\x1b[?1015l')
        self.write_raw('\x1b[?1006l')

    def erase_end_of_line(self):
        """
        Erases from the current cursor position to the end of the current line.
        """
        self.write_raw('\x1b[K')

    def erase_down(self):
        """
        Erases the screen from the current line down to the bottom of the
        screen.
        """
        self.write_raw('\x1b[J')

    def reset_attributes(self):
        self.write_raw('\x1b[0m')

    def set_attributes(self, attrs):
        """
        Create new style and output.

        :param attrs: `Attrs` instance.
        """
        if self.true_color() and (not self.ansi_colors_only()):
            self.write_raw(self._escape_code_cache_true_color[attrs])
        else:
            self.write_raw(self._escape_code_cache[attrs])

    def disable_autowrap(self):
        self.write_raw('\x1b[?7l')

    def enable_autowrap(self):
        self.write_raw('\x1b[?7h')

    def enable_bracketed_paste(self):
        self.write_raw('\x1b[?2004h')

    def disable_bracketed_paste(self):
        self.write_raw('\x1b[?2004l')

    def cursor_goto(self, row=0, column=0):
        """ Move cursor position. """
        self.write_raw('\x1b[%i;%iH' % (row, column))

    def cursor_up(self, amount):
        if amount == 0:
            pass
        elif amount == 1:
            self.write_raw('\x1b[A')
        else:
            self.write_raw('\x1b[%iA' % amount)

    def cursor_down(self, amount):
        if amount == 0:
            pass
        elif amount == 1:
            self.write_raw('\x1b[B')
        else:
            self.write_raw('\x1b[%iB' % amount)

    def cursor_forward(self, amount):
        if amount == 0:
            pass
        elif amount == 1:
            self.write_raw('\x1b[C')
        else:
            self.write_raw('\x1b[%iC' % amount)

    def cursor_backward(self, amount):
        if amount == 0:
            pass
        elif amount == 1:
            self.write_raw('\x08')
        else:
            self.write_raw('\x1b[%iD' % amount)

    def hide_cursor(self):
        self.write_raw('\x1b[?25l')

    def show_cursor(self):
        self.write_raw('\x1b[?12l\x1b[?25h')

    def flush(self):
        """
        Write to output stream and flush.
        """
        if not self._buffer:
            return
        data = ''.join(self._buffer)
        try:
            if self.write_binary:
                if hasattr(self.stdout, 'buffer'):
                    out = self.stdout.buffer
                else:
                    out = self.stdout
                out.write(data.encode(self.stdout.encoding or 'utf-8', 'replace'))
            else:
                self.stdout.write(data)
            self.stdout.flush()
        except IOError as e:
            if e.args and e.args[0] == errno.EINTR:
                pass
            elif e.args and e.args[0] == 0:
                pass
            else:
                raise
        self._buffer = []

    def ask_for_cpr(self):
        """
        Asks for a cursor position report (CPR).
        """
        self.write_raw('\x1b[6n')
        self.flush()

    def bell(self):
        """ Sound bell. """
        self.write_raw('\x07')
        self.flush()