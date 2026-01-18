import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def _input_prompt(self):
    result = [('Prompt', 'In['), ('PromptNum', '%d' % self.IP.execution_count), ('Prompt', ']: ')]
    self._prompt_size = sum((len(token[1]) for token in result))
    return result