import is finished. But that is no problem since the module is passed in.
import warnings
from textwrap import dedent
def _unused_fix_for_QtGui(QtGui):
    for name, cls in QtGui.__dict__.items():
        if name.startswith('QMatrix') and 'data' in cls.__dict__:
            cls.constData = constData