import os
import subprocess as sp
import tempfile
import warnings
import numpy as np
import proglog
from imageio import imread, imsave
from ..Clip import Clip
from ..compat import DEVNULL, string_types
from ..config import get_setting
from ..decorators import (add_mask_if_none, apply_to_mask,
from ..tools import (deprecated_version_of, extensions_dict, find_extension,
from .io.ffmpeg_writer import ffmpeg_write_video
from .io.gif_writers import (write_gif, write_gif_with_image_io,
from .tools.drawing import blit
class ImageClip(VideoClip):
    """Class for non-moving VideoClips.

    A video clip originating from a picture. This clip will simply
    display the given picture at all times.

    Examples
    ---------

    >>> clip = ImageClip("myHouse.jpeg")
    >>> clip = ImageClip( someArray ) # a Numpy array represent

    Parameters
    -----------

    img
      Any picture file (png, tiff, jpeg, etc.) or any array representing
      an RGB image (for instance a frame from a VideoClip).

    ismask
      Set this parameter to `True` if the clip is a mask.

    transparent
      Set this parameter to `True` (default) if you want the alpha layer
      of the picture (if it exists) to be used as a mask.

    Attributes
    -----------

    img
      Array representing the image of the clip.

    """

    def __init__(self, img, ismask=False, transparent=True, fromalpha=False, duration=None):
        VideoClip.__init__(self, ismask=ismask, duration=duration)
        if isinstance(img, string_types):
            img = imread(img)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                if fromalpha:
                    img = 1.0 * img[:, :, 3] / 255
                elif ismask:
                    img = 1.0 * img[:, :, 0] / 255
                elif transparent:
                    self.mask = ImageClip(1.0 * img[:, :, 3] / 255, ismask=True)
                    img = img[:, :, :3]
            elif ismask:
                img = 1.0 * img[:, :, 0] / 255
        self.make_frame = lambda t: img
        self.size = img.shape[:2][::-1]
        self.img = img

    def fl(self, fl, apply_to=None, keep_duration=True):
        """General transformation filter.

        Equivalent to VideoClip.fl . The result is no more an
        ImageClip, it has the class VideoClip (since it may be animated)
        """
        if apply_to is None:
            apply_to = []
        newclip = VideoClip.fl(self, fl, apply_to=apply_to, keep_duration=keep_duration)
        newclip.__class__ = VideoClip
        return newclip

    @outplace
    def fl_image(self, image_func, apply_to=None):
        """Image-transformation filter.

        Does the same as VideoClip.fl_image, but for ImageClip the
        tranformed clip is computed once and for all at the beginning,
        and not for each 'frame'.
        """
        if apply_to is None:
            apply_to = []
        arr = image_func(self.get_frame(0))
        self.size = arr.shape[:2][::-1]
        self.make_frame = lambda t: arr
        self.img = arr
        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.fl_image(image_func)
                setattr(self, attr, new_a)

    @outplace
    def fl_time(self, time_func, apply_to=None, keep_duration=False):
        """Time-transformation filter.

        Applies a transformation to the clip's timeline
        (see Clip.fl_time).

        This method does nothing for ImageClips (but it may affect their
        masks or their audios). The result is still an ImageClip.
        """
        if apply_to is None:
            apply_to = ['mask', 'audio']
        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.fl_time(time_func)
                setattr(self, attr, new_a)