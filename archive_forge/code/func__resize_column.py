import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _resize_column(self, event):
    """
        Callback used to resize a column of the table.  Return ``True``
        if the column is actually getting resized (if the user clicked
        on the far left or far right 5 pixels of a label); and
        ``False`` otherwies.
        """
    if event.widget.bind('<ButtonRelease>'):
        return False
    self._resize_column_index = None
    if event.widget is self:
        for i, lb in enumerate(self._listboxes):
            if abs(event.x - (lb.winfo_x() + lb.winfo_width())) < 10:
                self._resize_column_index = i
    elif event.x > event.widget.winfo_width() - 5:
        self._resize_column_index = event.widget.column_index
    elif event.x < 5 and event.widget.column_index != 0:
        self._resize_column_index = event.widget.column_index - 1
    if self._resize_column_index is not None:
        event.widget.bind('<Motion>', self._resize_column_motion_cb)
        event.widget.bind('<ButtonRelease-%d>' % event.num, self._resize_column_buttonrelease_cb)
        return True
    else:
        return False