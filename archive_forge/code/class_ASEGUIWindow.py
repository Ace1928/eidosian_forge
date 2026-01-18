import re
import sys
from collections import namedtuple
from functools import partial
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.messagebox import askokcancel as ask_question
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter.filedialog import LoadFileDialog, SaveFileDialog
from ase.gui.i18n import _
class ASEGUIWindow(MainWindow):

    def __init__(self, close, menu, config, scroll, scroll_event, press, move, release, resize):
        MainWindow.__init__(self, 'ASE-GUI', close, menu)
        self.size = np.array([450, 450])
        self.fg = config['gui_foreground_color']
        self.bg = config['gui_background_color']
        self.canvas = tk.Canvas(self.win, width=self.size[0], height=self.size[1], bg=self.bg, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.status = tk.Label(self.win, text='', anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        right = mouse_buttons.get(3, 3)
        self.canvas.bind('<ButtonPress>', bind(press))
        self.canvas.bind('<B1-Motion>', bind(move))
        self.canvas.bind('<B{right}-Motion>'.format(right=right), bind(move))
        self.canvas.bind('<ButtonRelease>', bind(release))
        self.canvas.bind('<Control-ButtonRelease>', bind(release, 'ctrl'))
        self.canvas.bind('<Shift-ButtonRelease>', bind(release, 'shift'))
        self.canvas.bind('<Configure>', resize)
        if not config['swap_mouse']:
            self.canvas.bind('<Shift-B{right}-Motion>'.format(right=right), bind(scroll))
        else:
            self.canvas.bind('<Shift-B1-Motion>', bind(scroll))
        self.win.bind('<MouseWheel>', bind(scroll_event))
        self.win.bind('<Key>', bind(scroll))
        self.win.bind('<Shift-Key>', bind(scroll, 'shift'))
        self.win.bind('<Control-Key>', bind(scroll, 'ctrl'))

    def update_status_line(self, text):
        self.status.config(text=text)

    def run(self):
        MainWindow.run(self)

    def click(self, name):
        self.callbacks[name]()

    def clear(self):
        self.canvas.delete(tk.ALL)

    def update(self):
        self.canvas.update_idletasks()

    def circle(self, color, selected, *bbox):
        if selected:
            outline = '#004500'
            width = 3
        else:
            outline = 'black'
            width = 1
        self.canvas.create_oval(*tuple((int(x) for x in bbox)), fill=color, outline=outline, width=width)

    def arc(self, color, selected, start, extent, *bbox):
        if selected:
            outline = '#004500'
            width = 3
        else:
            outline = 'black'
            width = 1
        self.canvas.create_arc(*tuple((int(x) for x in bbox)), start=start, extent=extent, fill=color, outline=outline, width=width)

    def line(self, bbox, width=1):
        self.canvas.create_line(*tuple((int(x) for x in bbox)), width=width)

    def text(self, x, y, txt, anchor=tk.CENTER, color='black'):
        anchor = {'SE': tk.SE}.get(anchor, anchor)
        self.canvas.create_text((x, y), text=txt, anchor=anchor, fill=color)

    def after(self, time, callback):
        id = self.win.after(int(time * 1000), callback)
        return namedtuple('Timer', 'cancel')(lambda: self.win.after_cancel(id))