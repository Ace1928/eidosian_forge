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
class BaseImageConverter(SphinxTransform):

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.findall(nodes.image):
            if self.match(node):
                self.handle(node)

    def match(self, node: nodes.image) -> bool:
        return True

    def handle(self, node: nodes.image) -> None:
        pass

    @property
    def imagedir(self) -> str:
        return os.path.join(self.app.doctreedir, 'images')