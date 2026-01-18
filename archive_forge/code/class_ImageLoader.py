import re
from base64 import b64decode
import imghdr
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.logger import Logger
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.atlas import Atlas
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.compat import string_types
from kivy.setupconfig import USE_SDL2
import zipfile
from io import BytesIO
from os import environ
from kivy.graphics.texture import Texture, TextureRegion
class ImageLoader(object):
    loaders = []

    @staticmethod
    def zip_loader(filename, **kwargs):
        """Read images from an zip file.

        .. versionadded:: 1.0.8

        Returns an Image with a list of type ImageData stored in Image._data
        """
        _file = BytesIO(open(filename, 'rb').read())
        z = zipfile.ZipFile(_file)
        image_data = []
        znamelist = z.namelist()
        znamelist.sort()
        image = None
        for zfilename in znamelist:
            try:
                tmpfile = BytesIO(z.read(zfilename))
                ext = zfilename.split('.')[-1].lower()
                im = None
                for loader in ImageLoader.loaders:
                    if ext not in loader.extensions() or not loader.can_load_memory():
                        continue
                    Logger.debug('Image%s: Load <%s> from <%s>' % (loader.__name__[11:], zfilename, filename))
                    try:
                        im = loader(zfilename, ext=ext, rawdata=tmpfile, inline=True, **kwargs)
                    except:
                        continue
                    break
                if im is not None:
                    image_data.append(im._data[0])
                    image = im
            except:
                Logger.warning('Image: Unable to load image<%s> in zip <%s> trying to continue...' % (zfilename, filename))
        z.close()
        if len(image_data) == 0:
            raise Exception('no images in zip <%s>' % filename)
        image._data = image_data
        image.filename = filename
        return image

    @staticmethod
    def register(defcls):
        ImageLoader.loaders.append(defcls)

    @staticmethod
    def load(filename, **kwargs):
        if filename[:8] == 'atlas://':
            rfn = filename[8:]
            try:
                rfn, uid = rfn.rsplit('/', 1)
            except ValueError:
                raise ValueError('Image: Invalid %s name for atlas' % filename)
            atlas = Cache.get('kv.atlas', rfn)
            if atlas:
                texture = atlas[uid]
                fn = 'atlas://%s/%s' % (rfn, uid)
                cid = '{}|{:d}|{:d}'.format(fn, False, 0)
                Cache.append('kv.texture', cid, texture)
                return Image(texture)
            afn = rfn
            if not afn.endswith('.atlas'):
                afn += '.atlas'
            afn = resource_find(afn)
            if not afn:
                raise Exception('Unable to find %r atlas' % afn)
            atlas = Atlas(afn)
            Cache.append('kv.atlas', rfn, atlas)
            for nid, texture in atlas.textures.items():
                fn = 'atlas://%s/%s' % (rfn, nid)
                cid = '{}|{:d}|{:d}'.format(fn, False, 0)
                Cache.append('kv.texture', cid, texture)
            return Image(atlas[uid])
        ext = filename.split('.')[-1].lower()
        if filename.startswith(('http://', 'https://')):
            ext = ext.split('?')[0]
        filename = resource_find(filename)
        if ext == 'zip':
            return ImageLoader.zip_loader(filename)
        else:
            im = None
            ext = imghdr.what(filename) or ext
            for loader in ImageLoader.loaders:
                if ext not in loader.extensions():
                    continue
                Logger.debug('Image%s: Load <%s>' % (loader.__name__[11:], filename))
                im = loader(filename, **kwargs)
                break
            if im is None:
                raise Exception('Unknown <%s> type, no loader found.' % ext)
            return im