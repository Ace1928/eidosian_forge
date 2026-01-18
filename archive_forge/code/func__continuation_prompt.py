import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def _continuation_prompt(self, size):
    prompt_text = ' ' * (size - 5) + '...: '
    return [('Prompt', prompt_text)]