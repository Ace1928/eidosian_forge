import sys
import os
import tkinter as tk
from tkinter import ttk
def _build_icons(self):
    if sys.platform == 'darwin':
        try:
            self.compress_icon = tk.Image('nsimage', source='NSExitFullScreenTemplate', width=18, height=18)
            self.expand_icon = tk.Image('nsimage', source='NSEnterFullScreenTemplate', width=18, height=18)
        except tk.TclError:
            self.compress_icon = tk.Image('photo', width=18, height=18, file=os.path.join(os.path.dirname(__file__), 'inward18.png'))
            self.expand_icon = tk.Image('photo', width=18, height=18, file=os.path.join(os.path.dirname(__file__), 'outward18.png'))
    else:
        suffix = 'gif' if tk.TkVersion < 8.6 else 'png'
        self.compress_icon = tk.Image('photo', width=18, height=18, file=os.path.join(os.path.dirname(__file__), 'inward18.' + suffix))
        self.expand_icon = tk.Image('photo', width=18, height=18, file=os.path.join(os.path.dirname(__file__), 'outward18.' + suffix))