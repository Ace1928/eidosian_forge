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
class LoaderBase(object):
    """Common base for the Loader and specific implementations.
    By default, the Loader will be the best available loader implementation.

    The _update() function is called every 1 / 25.s or each frame if we have
    less than 25 FPS.
    """
    _trigger_update = None
    "Alias for mimetype extensions.\n\n    If you have trouble to have the right extension to be detected,\n    you can either add #.EXT at the end of the url, or use this array\n    to correct the detection.\n    For example, a zip-file on Windows can be detected as pyz.\n\n    By default, '.pyz' is translated to '.zip'\n\n    .. versionadded:: 1.11.0\n    "
    EXT_ALIAS = {'.pyz': '.zip'}

    def __init__(self):
        self._loading_image = None
        self._error_image = None
        self._num_workers = 2
        self._max_upload_per_frame = 2
        self._paused = False
        self._resume_cond = threading.Condition()
        self._q_load = deque()
        self._q_done = deque()
        self._client = []
        self._running = False
        self._start_wanted = False
        self._trigger_update = Clock.create_trigger(self._update)

    def __del__(self):
        if self._trigger_update is not None:
            self._trigger_update.cancel()

    def _set_num_workers(self, num):
        if num < 2:
            raise Exception('Must have at least 2 workers')
        self._num_workers = num

    def _get_num_workers(self):
        return self._num_workers
    num_workers = property(_get_num_workers, _set_num_workers)
    'Number of workers to use while loading (used only if the loader\n    implementation supports it). This setting impacts the loader only on\n    initialization. Once the loader is started, the setting has no impact::\n\n        from kivy.loader import Loader\n        Loader.num_workers = 4\n\n    The default value is 2 for giving a smooth user experience. You could\n    increase the number of workers, then all the images will be loaded faster,\n    but the user will not been able to use the application while loading.\n    Prior to 1.6.0, the default number was 20, and loading many full-hd images\n    was completely blocking the application.\n\n    .. versionadded:: 1.6.0\n    '

    def _set_max_upload_per_frame(self, num):
        if num is not None and num < 1:
            raise Exception('Must have at least 1 image processing per image')
        self._max_upload_per_frame = num

    def _get_max_upload_per_frame(self):
        return self._max_upload_per_frame
    max_upload_per_frame = property(_get_max_upload_per_frame, _set_max_upload_per_frame)
    "The number of images to upload per frame. By default, we'll\n    upload only 2 images to the GPU per frame. If you are uploading many\n    small images, you can easily increase this parameter to 10 or more.\n    If you are loading multiple full HD images, the upload time may have\n    consequences and block the application. If you want a\n    smooth experience, use the default.\n\n    As a matter of fact, a Full-HD RGB image will take ~6MB in memory,\n    so it may take time. If you have activated mipmap=True too, then the\n    GPU must calculate the mipmap of these big images too, in real time.\n    Then it may be best to reduce the :attr:`max_upload_per_frame` to 1\n    or 2. If you want to get rid of that (or reduce it a lot), take a\n    look at the DDS format.\n\n    .. versionadded:: 1.6.0\n    "

    def _get_loading_image(self):
        if not self._loading_image:
            loading_png_fn = join(kivy_data_dir, 'images', 'image-loading.zip')
            self._loading_image = ImageLoader.load(filename=loading_png_fn)
        return self._loading_image

    def _set_loading_image(self, image):
        if isinstance(image, str):
            self._loading_image = ImageLoader.load(filename=image)
        else:
            self._loading_image = image
    loading_image = property(_get_loading_image, _set_loading_image)
    "Image used for loading.\n    You can change it by doing::\n\n        Loader.loading_image = 'loading.png'\n\n    .. versionchanged:: 1.6.0\n        Not readonly anymore.\n    "

    def _get_error_image(self):
        if not self._error_image:
            error_png_fn = join('atlas://data/images/defaulttheme/image-missing')
            self._error_image = ImageLoader.load(filename=error_png_fn)
        return self._error_image

    def _set_error_image(self, image):
        if isinstance(image, str):
            self._error_image = ImageLoader.load(filename=image)
        else:
            self._error_image = image
    error_image = property(_get_error_image, _set_error_image)
    "Image used for error.\n    You can change it by doing::\n\n        Loader.error_image = 'error.png'\n\n    .. versionchanged:: 1.6.0\n        Not readonly anymore.\n    "

    def start(self):
        """Start the loader thread/process."""
        self._running = True

    def run(self, *largs):
        """Main loop for the loader."""
        pass

    def stop(self):
        """Stop the loader thread/process."""
        self._running = False

    def pause(self):
        """Pause the loader, can be useful during interactions.

        .. versionadded:: 1.6.0
        """
        self._paused = True

    def resume(self):
        """Resume the loader, after a :meth:`pause`.

        .. versionadded:: 1.6.0
        """
        self._paused = False
        self._resume_cond.acquire()
        self._resume_cond.notify_all()
        self._resume_cond.release()

    def _wait_for_resume(self):
        while self._running and self._paused:
            self._resume_cond.acquire()
            self._resume_cond.wait(0.25)
            self._resume_cond.release()

    def _load(self, kwargs):
        """(internal) Loading function, called by the thread.
        Will call _load_local() if the file is local,
        or _load_urllib() if the file is on Internet.
        """
        while len(self._q_done) >= self.max_upload_per_frame * self._num_workers:
            sleep(0.1)
        self._wait_for_resume()
        filename = kwargs['filename']
        load_callback = kwargs['load_callback']
        post_callback = kwargs['post_callback']
        try:
            proto = filename.split(':', 1)[0]
        except:
            return
        if load_callback is not None:
            data = load_callback(filename)
        elif proto in ('http', 'https', 'ftp', 'smb'):
            data = self._load_urllib(filename, kwargs['kwargs'])
        else:
            data = self._load_local(filename, kwargs['kwargs'])
        if post_callback:
            data = post_callback(data)
        self._q_done.appendleft((filename, data))
        self._trigger_update()

    def _load_local(self, filename, kwargs):
        """(internal) Loading a local file"""
        return ImageLoader.load(filename, keep_data=True, **kwargs)

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

    def _update(self, *largs):
        """(internal) Check if a data is loaded, and pass to the client."""
        if self._start_wanted:
            if not self._running:
                self.start()
            self._start_wanted = False
        if self._paused:
            self._trigger_update()
            return
        for x in range(self.max_upload_per_frame):
            try:
                filename, data = self._q_done.pop()
            except IndexError:
                return
            image = data
            if not image.nocache:
                Cache.append('kv.loader', filename, image)
            for c_filename, client in self._client[:]:
                if filename != c_filename:
                    continue
                client.image = image
                client.loaded = True
                client.dispatch('on_load')
                self._client.remove((c_filename, client))
        self._trigger_update()

    def image(self, filename, load_callback=None, post_callback=None, **kwargs):
        """Load a image using the Loader. A ProxyImage is returned with a
        loading image. You can use it as follows::

            from kivy.app import App
            from kivy.uix.image import Image
            from kivy.loader import Loader

            class TestApp(App):
                def _image_loaded(self, proxyImage):
                    if proxyImage.image.texture:
                        self.image.texture = proxyImage.image.texture

                def build(self):
                    proxyImage = Loader.image("myPic.jpg")
                    proxyImage.bind(on_load=self._image_loaded)
                    self.image = Image()
                    return self.image

            TestApp().run()

        In order to cancel all background loading, call *Loader.stop()*.
        """
        data = Cache.get('kv.loader', filename)
        if data not in (None, False):
            return ProxyImage(data, loading_image=self.loading_image, loaded=True, **kwargs)
        client = ProxyImage(self.loading_image, loading_image=self.loading_image, **kwargs)
        self._client.append((filename, client))
        if data is None:
            self._q_load.appendleft({'filename': filename, 'load_callback': load_callback, 'post_callback': post_callback, 'kwargs': kwargs})
            if not kwargs.get('nocache', False):
                Cache.append('kv.loader', filename, False)
            self._start_wanted = True
            self._trigger_update()
        else:
            pass
        return client

    def remove_from_cache(self, filename):
        Cache.remove('kv.loader', filename)