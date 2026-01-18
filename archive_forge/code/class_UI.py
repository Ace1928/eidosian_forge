from __future__ import annotations
import tkinter
from io import BytesIO
from . import Image
class UI(tkinter.Label):

    def __init__(self, master, im):
        if im.mode == '1':
            self.image = BitmapImage(im, foreground='white', master=master)
        else:
            self.image = PhotoImage(im, master=master)
        super().__init__(master, image=self.image, bg='black', bd=0)