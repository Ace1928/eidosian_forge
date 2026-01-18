import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _resize_column_buttonrelease_cb(self, event):
    event.widget.unbind('<ButtonRelease-%d>' % event.num)
    event.widget.unbind('<Motion>')