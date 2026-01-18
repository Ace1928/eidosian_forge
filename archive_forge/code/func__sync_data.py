from __future__ import annotations
from base64 import b64encode
from pathlib import Path
from typing import (
import param
from param.parameterized import eval_function_with_deps, iscoroutinefunction
from pyviz_comms import JupyterComm
from ..io.notebook import push
from ..io.resources import CDN_DIST
from ..io.state import state
from ..models import (
from ..util import lazy_load
from .base import Widget
from .button import BUTTON_STYLES, BUTTON_TYPES, IconMixin
from .indicators import Progress  # noqa
def _sync_data(self, fileobj):
    filename = self.filename
    if isinstance(fileobj, (str, Path)):
        fileobj = Path(fileobj)
        if not fileobj.exists():
            raise FileNotFoundError('File "%s" not found.' % fileobj)
        with open(fileobj, 'rb') as f:
            b64 = b64encode(f.read()).decode('utf-8')
        if filename is None:
            filename = fileobj.name
    elif hasattr(fileobj, 'read'):
        bdata = fileobj.read()
        if not isinstance(bdata, bytes):
            bdata = bdata.encode('utf-8')
        b64 = b64encode(bdata).decode('utf-8')
        if filename is None:
            raise ValueError('Must provide filename if file-like object is provided.')
    else:
        raise ValueError('Cannot transfer unknown object of type %s' % type(fileobj).__name__)
    ext = filename.split('.')[-1]
    stype, mtype = (None, None)
    for mime_type, subtypes in self._mime_types.items():
        if ext in subtypes:
            mtype = mime_type
            stype = subtypes[ext]
            break
    if stype is None:
        mime = 'application/octet-stream'
    else:
        mime = '{type}/{subtype}'.format(type=mtype, subtype=stype)
    data = 'data:{mime};base64,{b64}'.format(mime=mime, b64=b64)
    self._synced = True
    self.param.update(data=data, filename=filename)
    self._update_label()
    self._transfers += 1