import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
class LaxTemplate(string.Template):
    pattern = '\n    \\$(?:\n      (?P<escaped>\\$)             |   # Escape sequence of two delimiters\n      (?P<named>[_a-z][_a-z0-9]*) |   # delimiter and a Python identifier\n      {(?P<braced>.*?)}           |   # delimiter and a braced identifier\n      (?P<invalid>)                   # Other ill-formed delimiter exprs\n    )\n    '