import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
class FpsLabelUpdater:

    def __init__(self, label):
        self.label = label
        self.num_iterations = 0
        self.total_time = 0.0
        self.last_time = time.time()

    def __call__(self, t):
        self.num_iterations += 1
        self.total_time += t
        if self.num_iterations > 50 or time.time() - self.last_time > 2.0:
            current_time = time.time()
            fps = self.num_iterations / (current_time - self.last_time)
            time_ms = 1000 * self.total_time / self.num_iterations
            self.label.configure(text='%.1ffps (%dms)' % (fps, time_ms))
            self.last_time = current_time
            self.num_iterations = 0
            self.total_time = 0.0