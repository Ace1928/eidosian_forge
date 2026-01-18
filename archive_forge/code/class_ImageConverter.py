import os
import re
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import epoch_to_rfc1123, logging, requests, rfc1123_to_epoch, sha1
from sphinx.util.images import get_image_extension, guess_mimetype, parse_data_uri
from sphinx.util.osutil import ensuredir
class ImageConverter(BaseImageConverter):
    """A base class for image converters.

    An image converter is kind of Docutils transform module.  It is used to
    convert image files which are not supported by a builder to the
    appropriate format for that builder.

    For example, :py:class:`LaTeX builder <.LaTeXBuilder>` supports PDF,
    PNG and JPEG as image formats.  However it does not support SVG images.
    For such case, using image converters allows to embed these
    unsupported images into the document.  One of the image converters;
    :ref:`sphinx.ext.imgconverter <sphinx.ext.imgconverter>` can convert
    a SVG image to PNG format using Imagemagick internally.

    There are three steps to make your custom image converter:

    1. Make a subclass of ``ImageConverter`` class
    2. Override ``conversion_rules``, ``is_available()`` and ``convert()``
    3. Register your image converter to Sphinx using
       :py:meth:`.Sphinx.add_post_transform`
    """
    default_priority = 200
    available: Optional[bool] = None
    conversion_rules: List[Tuple[str, str]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def match(self, node: nodes.image) -> bool:
        if not self.app.builder.supported_image_types:
            return False
        elif '?' in node['candidates']:
            return False
        elif set(self.guess_mimetypes(node)) & set(self.app.builder.supported_image_types):
            return False
        elif self.available is None:
            self.__class__.available = self.is_available()
        if not self.available:
            return False
        else:
            rule = self.get_conversion_rule(node)
            if rule:
                return True
            else:
                return False

    def get_conversion_rule(self, node: nodes.image) -> Tuple[str, str]:
        for candidate in self.guess_mimetypes(node):
            for supported in self.app.builder.supported_image_types:
                rule = (candidate, supported)
                if rule in self.conversion_rules:
                    return rule
        return None

    def is_available(self) -> bool:
        """Return the image converter is available or not."""
        raise NotImplementedError()

    def guess_mimetypes(self, node: nodes.image) -> List[str]:
        if '?' in node['candidates']:
            return []
        elif '*' in node['candidates']:
            return [guess_mimetype(node['uri'])]
        else:
            return node['candidates'].keys()

    def handle(self, node: nodes.image) -> None:
        _from, _to = self.get_conversion_rule(node)
        if _from in node['candidates']:
            srcpath = node['candidates'][_from]
        else:
            srcpath = node['candidates']['*']
        filename = get_filename_for(srcpath, _to)
        ensuredir(self.imagedir)
        destpath = os.path.join(self.imagedir, filename)
        abs_srcpath = os.path.join(self.app.srcdir, srcpath)
        if self.convert(abs_srcpath, destpath):
            if '*' in node['candidates']:
                node['candidates']['*'] = destpath
            else:
                node['candidates'][_to] = destpath
            node['uri'] = destpath
            self.env.original_image_uri[destpath] = srcpath
            self.env.images.add_file(self.env.docname, destpath)

    def convert(self, _from: str, _to: str) -> bool:
        """Convert an image file to the expected format.

        *_from* is a path of the source image file, and *_to* is a path
        of the destination file.
        """
        raise NotImplementedError()