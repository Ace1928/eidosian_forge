import parlai.core.build_data as build_data
import parlai.utils.logging as logging
import os
from PIL import Image
from zipfile import ZipFile
def _image_mode_switcher(self):
    if self.image_mode not in IMAGE_MODE_SWITCHER:
        raise NotImplementedError('image preprocessing mode' + '{} not supported yet'.format(self.image_mode))
    return IMAGE_MODE_SWITCHER.get(self.image_mode)