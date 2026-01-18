import inspect
import re
import six
def _lines_walker(self, lines, start_pos):
    """
        Using the curses urwid library, displays all lines passed as argument,
        and after allowing selection of one line using up, down and enter keys,
        returns its index.
        @param lines: The lines to display and select from.
        @type lines: list of str
        @param start_pos: The index of the line to select initially.
        @type start_pos: int
        @return: the index of the selected line.
        @rtype: int
        """
    import urwid
    palette = [('header', 'white', 'black'), ('reveal focus', 'black', 'yellow', 'standout')]
    content = urwid.SimpleListWalker([urwid.AttrMap(w, None, 'reveal focus') for w in [urwid.Text(line) for line in lines]])
    listbox = urwid.ListBox(content)
    frame = urwid.Frame(listbox)

    def handle_input(input, raw):
        for key in input:
            widget, pos = content.get_focus()
            if key == 'up':
                if pos > 0:
                    content.set_focus(pos - 1)
            elif key == 'down':
                try:
                    content.set_focus(pos + 1)
                except IndexError:
                    pass
            elif key == 'enter':
                raise urwid.ExitMainLoop()
    content.set_focus(start_pos)
    loop = urwid.MainLoop(frame, palette, input_filter=handle_input)
    loop.run()
    return listbox.focus_position