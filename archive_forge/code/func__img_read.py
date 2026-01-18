from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader
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