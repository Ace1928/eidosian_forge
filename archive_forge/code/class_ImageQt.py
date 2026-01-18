from __future__ import annotations
import sys
from io import BytesIO
from . import Image
from ._util import is_path
class ImageQt(QImage):

    def __init__(self, im):
        """
            An PIL image wrapper for Qt.  This is a subclass of PyQt's QImage
            class.

            :param im: A PIL Image object, or a file name (given either as
                Python string or a PyQt string object).
            """
        im_data = _toqclass_helper(im)
        self.__data = im_data['data']
        super().__init__(self.__data, im_data['size'][0], im_data['size'][1], im_data['format'])
        if im_data['colortable']:
            self.setColorTable(im_data['colortable'])