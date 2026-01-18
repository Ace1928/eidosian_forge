from __future__ import annotations
import asyncio
import base64
import struct
from io import BytesIO
from pathlib import PurePath
from typing import (
import param
from ..models import PDF as _BkPDF
from ..util import isfile, isurl
from .markup import HTMLBasePane, escape
class SVG(ImageBase):
    """
    The `SVG` pane embeds a .svg image file in a panel if provided a
    local path, or will link to a remote image if provided a URL.

    Reference: https://panel.holoviz.org/reference/panes/SVG.html

    :Example:

    >>> SVG(
    ...     'https://upload.wikimedia.org/wikipedia/commons/6/6b/Bitmap_VS_SVG.svg',
    ...     alt_text='A gif vs svg comparison',
    ...     link_url='https://en.wikipedia.org/wiki/SVG',
    ...     width=300, height=400
    ... )
    """
    encode = param.Boolean(default=True, doc='\n        Whether to enable base64 encoding of the SVG, base64 encoded\n        SVGs do not support links.')
    filetype: ClassVar[str] = 'svg+xml'
    _rename: ClassVar[Mapping[str, str | None]] = {'encode': None}
    _rerender_params: ClassVar[List[str]] = ImageBase._rerender_params + ['encode']

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        return super().applies(obj) or (isinstance(obj, str) and obj.lstrip().startswith('<svg'))

    def _type_error(self, object):
        if isinstance(object, str):
            raise ValueError('%s pane cannot parse string that is not a filename, URL or a SVG XML contents.' % type(self).__name__)
        super()._type_error(object)

    def _data(self, obj):
        if isinstance(obj, str) and obj.lstrip().startswith('<svg'):
            return obj
        return super()._data(obj)

    def _imgshape(self, data):
        return (self.width, self.height)

    def _transform_object(self, obj: Any) -> Dict[str, Any]:
        width, height = (self.width, self.height)
        w, h = self._img_dims(width, height)
        if self.embed or (isfile(obj) or (isinstance(obj, str) and obj.lstrip().startswith('<svg')) or (not isinstance(obj, (str, PurePath)))):
            data = self._data(obj)
        else:
            return dict(object=self._format_html(obj, w, h))
        if data is None:
            return dict(object='<img></img>')
        if self.encode:
            ws = f' width: {w};' if w else ''
            hs = f' height: {h};' if h else ''
            object_fit = 'contain' if self.fixed_aspect else 'fill'
            data = f'<img src="{self._b64(data)}" style="max-width: 100%; max-height: 100%; object-fit: {object_fit};{ws}{hs}"></img>'
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return dict(width=width, height=height, text=escape(data))