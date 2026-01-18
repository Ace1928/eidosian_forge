import locale as pylocale
import unicodedata
import warnings
import numpy as np
from onnx.reference.op_run import OpRun, RuntimeTypeError
@staticmethod
def _remove_stopwords(text, stops):
    spl = text.split(' ')
    return ' '.join(filter(lambda s: s not in stops, spl))