from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from prompt_toolkit.layout.screen import Size
class DummyOutput(Output):
    """
    For testing. An output class that doesn't render anything.
    """

    def fileno(self):
        """ There is no sensible default for fileno(). """
        raise NotImplementedError

    def encoding(self):
        return 'utf-8'

    def write(self, data):
        pass

    def write_raw(self, data):
        pass

    def set_title(self, title):
        pass

    def clear_title(self):
        pass

    def flush(self):
        pass

    def erase_screen(self):
        pass

    def enter_alternate_screen(self):
        pass

    def quit_alternate_screen(self):
        pass

    def enable_mouse_support(self):
        pass

    def disable_mouse_support(self):
        pass

    def erase_end_of_line(self):
        pass

    def erase_down(self):
        pass

    def reset_attributes(self):
        pass

    def set_attributes(self, attrs):
        pass

    def disable_autowrap(self):
        pass

    def enable_autowrap(self):
        pass

    def cursor_goto(self, row=0, column=0):
        pass

    def cursor_up(self, amount):
        pass

    def cursor_down(self, amount):
        pass

    def cursor_forward(self, amount):
        pass

    def cursor_backward(self, amount):
        pass

    def hide_cursor(self):
        pass

    def show_cursor(self):
        pass

    def ask_for_cpr(self):
        pass

    def bell(self):
        pass

    def enable_bracketed_paste(self):
        pass

    def disable_bracketed_paste(self):
        pass

    def get_size(self):
        return Size(rows=40, columns=80)