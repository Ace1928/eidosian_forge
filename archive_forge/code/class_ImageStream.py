from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class ImageStream(MediaStream):
    """Represent a media stream by a static image"""
    _model_name = Unicode('ImageStreamModel').tag(sync=True)
    image = Instance(Image, help='An ipywidgets.Image instance that will be the source of the media stream.').tag(sync=True, **widget_serialization)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Create a `ImageStream` from a local file.

        Parameters
        ----------
        filename: str
            The location of a file to read into the value from disk.
        **kwargs
            Extra keyword arguments for `ImageStream`
        """
        return cls(image=Image.from_file(filename), **kwargs)

    @classmethod
    def from_url(cls, url, **kwargs):
        """Create a `ImageStream` from a url.

        This will create a `ImageStream` from an Image using its url

        Parameters
        ----------
        url: str
            The url of the file that will be used for the .image trait.
        **kwargs
            Extra keyword arguments for `ImageStream`
        """
        return cls(image=Image.from_url(url), **kwargs)

    @classmethod
    def from_download(cls, url, **kwargs):
        """Create a `ImageStream` from a url by downloading

        Parameters
        ----------
        url: str
            The url of the file that will be downloadeded and its bytes
            assigned to the value trait of the video trait.
        **kwargs
            Extra keyword arguments for `ImageStream`
        """
        ext = os.path.splitext(url)[1]
        if ext:
            format = ext[1:]
        image = Image(value=urlopen(url).read(), format=format)
        return cls(image=image, **kwargs)