import sys
import os
import tkinter as tk
from tkinter import ttk
def _update_labels(self):
    l = self.slider.left_end
    r = self.slider.right_end
    num_digits = _num_digits(r - l)
    format_str1 = '%%.%df' % (num_digits + 1)
    format_str2 = '%%.%df' % num_digits
    self.value_label.configure(text=format_str1 % self.current_value)
    self.min_label.configure(text=format_str2 % l)
    self.max_label.configure(text=format_str2 % r)
    if self.on_change:
        self.on_change()