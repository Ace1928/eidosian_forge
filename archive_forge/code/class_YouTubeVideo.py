from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
class YouTubeVideo(IFrame):
    """Class for embedding a YouTube Video in an IPython session, based on its video id.

    e.g. to embed the video from https://www.youtube.com/watch?v=foo , you would
    do::

        vid = YouTubeVideo("foo")
        display(vid)

    To start from 30 seconds::

        vid = YouTubeVideo("abc", start=30)
        display(vid)

    To calculate seconds from time as hours, minutes, seconds use
    :class:`datetime.timedelta`::

        start=int(timedelta(hours=1, minutes=46, seconds=40).total_seconds())

    Other parameters can be provided as documented at
    https://developers.google.com/youtube/player_parameters#Parameters
    
    When converting the notebook using nbconvert, a jpeg representation of the video
    will be inserted in the document.
    """

    def __init__(self, id, width=400, height=300, allow_autoplay=False, **kwargs):
        self.id = id
        src = 'https://www.youtube.com/embed/{0}'.format(id)
        if allow_autoplay:
            extras = list(kwargs.get('extras', [])) + ['allow="autoplay"']
            kwargs.update(autoplay=1, extras=extras)
        super(YouTubeVideo, self).__init__(src, width, height, **kwargs)

    def _repr_jpeg_(self):
        from urllib.request import urlopen
        try:
            return urlopen('https://img.youtube.com/vi/{id}/hqdefault.jpg'.format(id=self.id)).read()
        except IOError:
            return None