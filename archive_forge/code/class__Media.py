import mimetypes
from .widget_core import CoreWidget
from .domwidget import DOMWidget
from .valuewidget import ValueWidget
from .widget import register
from traitlets import Unicode, CUnicode, Bool
from .trait_types import CByteMemoryView
@register
class _Media(DOMWidget, ValueWidget, CoreWidget):
    """Base class for Image, Audio and Video widgets.

    The `value` of this widget accepts a byte string.  The byte string is the
    raw data that you want the browser to display.

    If you pass `"url"` to the `"format"` trait, `value` will be interpreted
    as a URL as bytes encoded in UTF-8.
    """
    value = CByteMemoryView(help='The media data as a memory view of bytes.').tag(sync=True)

    @classmethod
    def _from_file(cls, tag, filename, **kwargs):
        """
        Create an :class:`Media` from a local file.

        Parameters
        ----------
        filename: str
            The location of a file to read into the value from disk.

        **kwargs:
            The keyword arguments for `Media`

        Returns an `Media` with the value set from the filename.
        """
        value = cls._load_file_value(filename)
        if 'format' not in kwargs:
            format = cls._guess_format(tag, filename)
            if format is not None:
                kwargs['format'] = format
        return cls(value=value, **kwargs)

    @classmethod
    def from_url(cls, url, **kwargs):
        """
        Create an :class:`Media` from a URL.

        :code:`Media.from_url(url)` is equivalent to:

        .. code-block: python

            med = Media(value=url, format='url')

        But both unicode and bytes arguments are allowed for ``url``.

        Parameters
        ----------
        url: [str, bytes]
            The location of a URL to load.
        """
        if isinstance(url, str):
            url = url.encode('utf-8')
        return cls(value=url, format='url', **kwargs)

    def set_value_from_file(self, filename):
        """
        Convenience method for reading a file into `value`.

        Parameters
        ----------
        filename: str
            The location of a file to read into value from disk.
        """
        value = self._load_file_value(filename)
        self.value = value

    @classmethod
    def _load_file_value(cls, filename):
        if getattr(filename, 'read', None) is not None:
            return filename.read()
        else:
            with open(filename, 'rb') as f:
                return f.read()

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

    def _get_repr(self, cls):
        class_name = self.__class__.__name__
        signature = []
        sig_value = 'value={!r}'.format(self.value[:40].tobytes())
        if self.value.nbytes > 40:
            sig_value = sig_value[:-1] + '...' + sig_value[-1]
        signature.append(sig_value)
        for key in super(cls, self)._repr_keys():
            if key == 'value':
                continue
            value = str(getattr(self, key))
            signature.append('{}={!r}'.format(key, value))
        signature = ', '.join(signature)
        return '{}({})'.format(class_name, signature)