from __future__ import annotations
from . import Image, ImageFilter, ImageStat
class Sharpness(_Enhance):
    """Adjust image sharpness.

    This class can be used to adjust the sharpness of an image. An
    enhancement factor of 0.0 gives a blurred image, a factor of 1.0 gives the
    original image, and a factor of 2.0 gives a sharpened image.
    """

    def __init__(self, image):
        self.image = image
        self.degenerate = image.filter(ImageFilter.SMOOTH)
        if 'A' in image.getbands():
            self.degenerate.putalpha(image.getchannel('A'))