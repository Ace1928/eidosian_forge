from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader
class ImageLoaderPIL(ImageLoaderBase):
    """Image loader based on the PIL library.

    .. versionadded:: 1.0.8

    Support for GIF animation added.

    Gif animation has a lot of issues(transparency/color depths... etc).
    In order to keep it simple, what is implemented here is what is
    natively supported by the PIL library.

    As a general rule, try to use gifs that have no transparency.
    Gif's with transparency will work but be prepared for some
    artifacts until transparency support is improved.

    """

    @staticmethod
    def can_save(fmt, is_bytesio):
        if is_bytesio:
            return False
        return fmt in ImageLoaderPIL.extensions()

    @staticmethod
    def can_load_memory():
        return True

    @staticmethod
    def extensions():
        """Return accepted extensions for this loader"""
        PILImage.init()
        return tuple((ext_with_dot[1:] for ext_with_dot in PILImage.EXTENSION))

    def _img_correct(self, _img_tmp):
        """Convert image to the correct format and orientation.
        """
        if _img_tmp.mode.lower() not in ('rgb', 'rgba'):
            try:
                imc = _img_tmp.convert('RGBA')
            except:
                Logger.warning('Image: Unable to convert image to rgba (was %s)' % _img_tmp.mode.lower())
                raise
            _img_tmp = imc
        return _img_tmp

    def _img_read(self, im):
        """Read images from an animated file.
        """
        im.seek(0)
        try:
            img_ol = None
            while True:
                img_tmp = im
                img_tmp = self._img_correct(img_tmp)
                if img_ol and (hasattr(im, 'dispose') and (not im.dispose)):
                    img_ol.paste(img_tmp, (0, 0), img_tmp)
                    img_tmp = img_ol
                img_ol = img_tmp
                yield ImageData(img_tmp.size[0], img_tmp.size[1], img_tmp.mode.lower(), img_tmp.tobytes())
                im.seek(im.tell() + 1)
        except EOFError:
            pass

    def load(self, filename):
        try:
            im = PILImage.open(filename)
        except:
            Logger.warning('Image: Unable to load image <%s>' % filename)
            raise
        if not self._inline:
            self.filename = filename
        return list(self._img_read(im))

    @staticmethod
    def save(filename, width, height, pixelfmt, pixels, flipped=False, imagefmt=None):
        image = PILImage.frombytes(pixelfmt.upper(), (width, height), pixels)
        if flipped:
            image = image.transpose(PILImage.FLIP_TOP_BOTTOM)
        image.save(filename)
        return True