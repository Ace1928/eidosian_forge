import sys
import os
import tkinter as tk
from tkinter import ttk
def _handle_wheel(self, event):
    delta = event.delta
    if delta < 0:
        self.compresser.state(['pressed'])
        while delta < 0:
            self.zoom_in()
            delta += 1
        self.after(200, lambda: self.compresser.state(['!pressed']))
    elif delta > 0:
        self.expander.state(['pressed'])
        while delta > 0:
            self.zoom_out()
            delta -= 1
        self.after(200, lambda: self.expander.state(['!pressed']))