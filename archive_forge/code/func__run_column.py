import locale as pylocale
import unicodedata
import warnings
import numpy as np
from onnx.reference.op_run import OpRun, RuntimeTypeError
@staticmethod
def _run_column(cin, cout, slocale=None, stops=None, raw_stops=None, is_case_sensitive=None, case_change_action=None):
    if pylocale.getlocale() != slocale:
        try:
            pylocale.setlocale(pylocale.LC_ALL, slocale)
        except pylocale.Error as e:
            warnings.warn(f'Unknown local setting {slocale!r} (current: {pylocale.getlocale()!r}) - {e!r}.', stacklevel=1)
    cout[:] = cin[:]
    for i in range(0, cin.shape[0]):
        if isinstance(cout[i], float):
            cout[i] = ''
        else:
            cout[i] = StringNormalizer.strip_accents_unicode(cout[i])
    if is_case_sensitive and len(stops) > 0:
        for i in range(0, cin.shape[0]):
            cout[i] = StringNormalizer._remove_stopwords(cout[i], raw_stops)
    if case_change_action == 'LOWER':
        for i in range(0, cin.shape[0]):
            cout[i] = cout[i].lower()
    elif case_change_action == 'UPPER':
        for i in range(0, cin.shape[0]):
            cout[i] = cout[i].upper()
    elif case_change_action != 'NONE':
        raise RuntimeError(f'Unknown option for case_change_action: {case_change_action!r}.')
    if not is_case_sensitive and len(stops) > 0:
        for i in range(0, cin.shape[0]):
            cout[i] = StringNormalizer._remove_stopwords(cout[i], stops)
    return cout