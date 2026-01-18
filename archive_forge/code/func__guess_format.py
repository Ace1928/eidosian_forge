import mimetypes
from .widget_core import CoreWidget
from .domwidget import DOMWidget
from .valuewidget import ValueWidget
from .widget import register
from traitlets import Unicode, CUnicode, Bool
from .trait_types import CByteMemoryView
@classmethod
def _guess_format(cls, tag, filename):
    name = getattr(filename, 'name', None)
    name = name or filename
    try:
        mtype, _ = mimetypes.guess_type(name)
        if not mtype.startswith('{}/'.format(tag)):
            return None
        return mtype[len('{}/'.format(tag)):]
    except Exception:
        return None