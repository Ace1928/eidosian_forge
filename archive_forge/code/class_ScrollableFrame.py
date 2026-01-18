import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
class ScrollableFrame(ttk.Frame):

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.header = header = ttk.Frame(self)
        header.pack(anchor='sw')
        self.canvas = canvas = tkinter.Canvas(self)
        self.scrollbar = scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind('<Configure>', self.resize)
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=self.set_scrollbar)
        canvas.pack(side='left', fill='both', expand=True, anchor='nw', pady=10)
        scrollbar.pack(side='right', fill='y', anchor='nw', pady=10)
        scrollbar.is_visible = True
        self.num_columns = 0
        self.has_mouse = False
        self.bind('<Enter>', self.mouse_in)
        self.bind('<Leave>', self.mouse_out)
        self.bind_all('<MouseWheel>', self.mouse_wheel)

    def headings(self, columninfo):
        for heading, column, weight, span in columninfo:
            self.num_columns = max(self.num_columns, 1 + column)
            self.header.columnconfigure(column, weight=weight)
            self.scrollable_frame.columnconfigure(column, weight=weight)
            ttk.Label(self.header, text=heading).grid(row=0, column=column, columnspan=span)

    def set_widths(self):
        for n in range(self.num_columns):
            header_width = self.header.grid_bbox(n, 0, n, 0)[2]
            column_width = self.scrollable_frame.grid_bbox(n, 0, n, 0)[2]
            width = max(header_width, column_width, 40)
            self.header.columnconfigure(n, minsize=width)
            self.scrollable_frame.columnconfigure(n, minsize=width)

    def set_scrollbar(self, low, high):
        if float(low) <= 0.0 and float(high) >= 1.0:
            self.scrollbar.pack_forget()
            self.scrollbar.is_visible = False
        else:
            self.scrollbar.pack(side='right', fill='y', anchor='nw', pady=10)
            self.scrollbar.is_visible = True
        self.scrollbar.set(low, high)

    def resize(self, event=None):
        self.set_widths()
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def mouse_in(self, event=None):
        self.has_mouse = True

    def mouse_out(self, event=None):
        self.has_mouse = False

    def mouse_wheel(self, event=None):
        if not self.has_mouse or not self.scrollbar.is_visible:
            return
        low, high = self.scrollbar.get()
        delta = event.delta
        self.canvas.yview_scroll(-delta, 'units')