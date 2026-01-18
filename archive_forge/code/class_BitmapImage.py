from __future__ import annotations
import tkinter
from io import BytesIO
from . import Image
class BitmapImage:
    """
    A Tkinter-compatible bitmap image.  This can be used everywhere Tkinter
    expects an image object.

    The given image must have mode "1".  Pixels having value 0 are treated as
    transparent.  Options, if any, are passed on to Tkinter.  The most commonly
    used option is ``foreground``, which is used to specify the color for the
    non-transparent parts.  See the Tkinter documentation for information on
    how to specify colours.

    :param image: A PIL image.
    """

    def __init__(self, image=None, **kw):
        if image is None:
            image = _get_image_from_kw(kw)
        self.__mode = image.mode
        self.__size = image.size
        if _pilbitmap_check():
            image.load()
            kw['data'] = f'PIL:{image.im.id}'
            self.__im = image
        else:
            kw['data'] = image.tobitmap()
        self.__photo = tkinter.BitmapImage(**kw)

    def __del__(self):
        name = self.__photo.name
        self.__photo.name = None
        try:
            self.__photo.tk.call('image', 'delete', name)
        except Exception:
            pass

    def width(self):
        """
        Get the width of the image.

        :return: The width, in pixels.
        """
        return self.__size[0]

    def height(self):
        """
        Get the height of the image.

        :return: The height, in pixels.
        """
        return self.__size[1]

    def __str__(self):
        """
        Get the Tkinter bitmap image identifier.  This method is automatically
        called by Tkinter whenever a BitmapImage object is passed to a Tkinter
        method.

        :return: A Tkinter bitmap image identifier (a string).
        """
        return str(self.__photo)