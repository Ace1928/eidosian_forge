from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader
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