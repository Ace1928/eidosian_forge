from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
class VimeoVideo(IFrame):
    """
    Class for embedding a Vimeo video in an IPython session, based on its video id.
    """

    def __init__(self, id, width=400, height=300, **kwargs):
        src = 'https://player.vimeo.com/video/{0}'.format(id)
        super(VimeoVideo, self).__init__(src, width, height, **kwargs)