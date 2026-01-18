import decimal as _decimal
import math as _math
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from io import StringIO as UnicodeIO
from types import SimpleNamespace
from .textTools import Tag, bytechr, byteord, bytesjoin, strjoin, tobytes, tostr
class Py23Error(NotImplementedError):
    pass