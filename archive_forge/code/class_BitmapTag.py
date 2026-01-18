import os
import zlib
import time  # noqa
import logging
import numpy as np
class BitmapTag(DefinitionTag):

    def __init__(self, im):
        DefinitionTag.__init__(self)
        self.tagtype = 36
        if len(im.shape) == 3:
            if im.shape[2] in [3, 4]:
                tmp = np.ones((im.shape[0], im.shape[1], 4), dtype=np.uint8) * 255
                for i in range(3):
                    tmp[:, :, i + 1] = im[:, :, i]
                if im.shape[2] == 4:
                    tmp[:, :, 0] = im[:, :, 3]
            else:
                raise ValueError('Invalid shape to be an image.')
        elif len(im.shape) == 2:
            tmp = np.ones((im.shape[0], im.shape[1], 4), dtype=np.uint8) * 255
            for i in range(3):
                tmp[:, :, i + 1] = im[:, :]
        else:
            raise ValueError('Invalid shape to be an image.')
        self._data = zlib.compress(tmp.tobytes(), zlib.DEFLATED)
        self.imshape = im.shape

    def process_tag(self):
        bb = bytes()
        bb += int2uint16(self.id)
        bb += int2uint8(5)
        bb += int2uint16(self.imshape[1])
        bb += int2uint16(self.imshape[0])
        bb += self._data
        self.bytes = bb