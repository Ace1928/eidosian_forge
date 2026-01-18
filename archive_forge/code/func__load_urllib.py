from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.core.image import ImageLoader, Image
from kivy.config import Config
from kivy.utils import platform
from collections import deque
from time import sleep
from os.path import join
from os import write, close, unlink, environ
import threading
import mimetypes
def _load_urllib(self, filename, kwargs):
    """(internal) Loading a network file. First download it, save it to a
        temporary file, and pass it to _load_local()."""
    import urllib.request
    import tempfile
    proto = filename.split(':', 1)[0]
    if proto == 'smb':
        try:
            from smb.SMBHandler import SMBHandler
        except ImportError:
            Logger.warning('Loader: can not load PySMB: make sure it is installed')
            return
    data = fd = _out_osfd = None
    try:
        _out_filename = ''
        if proto == 'smb':
            fd = urllib.request.build_opener(SMBHandler).open(filename)
        else:
            request = urllib.request.Request(filename)
            if Config.has_option('network', 'useragent'):
                useragent = Config.get('network', 'useragent')
                if useragent:
                    request.add_header('User-Agent', useragent)
            ssl_ctx = None
            if platform in ['android', 'ios']:
                import certifi
                import ssl
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            fd = urllib.request.urlopen(request, context=ssl_ctx)
        if '#.' in filename:
            suffix = '.' + filename.split('#.')[-1]
        else:
            ctype = fd.info().get_content_type()
            suffix = mimetypes.guess_extension(ctype)
            suffix = LoaderBase.EXT_ALIAS.get(suffix, suffix)
            if not suffix:
                parts = filename.split('?')[0].split('/')[1:]
                while len(parts) > 1 and (not parts[0]):
                    parts = parts[1:]
                if len(parts) > 1 and '.' in parts[-1]:
                    suffix = '.' + parts[-1].split('.')[-1]
        _out_osfd, _out_filename = tempfile.mkstemp(prefix='kivyloader', suffix=suffix)
        idata = fd.read()
        fd.close()
        fd = None
        write(_out_osfd, idata)
        close(_out_osfd)
        _out_osfd = None
        data = self._load_local(_out_filename, kwargs)
        for imdata in data._data:
            imdata.source = filename
    except Exception as ex:
        Logger.exception('Loader: Failed to load image <%s>' % filename)
        try:
            if _out_osfd:
                close(_out_osfd)
        except OSError:
            pass
        for c_filename, client in self._client[:]:
            if filename != c_filename:
                continue
            client.image = self.error_image
            client.dispatch('on_error', error=ex)
            self._client.remove((c_filename, client))
        return self.error_image
    finally:
        if fd:
            fd.close()
        if _out_osfd:
            close(_out_osfd)
        if _out_filename != '':
            unlink(_out_filename)
    return data