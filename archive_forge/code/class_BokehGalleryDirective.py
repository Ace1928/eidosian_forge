from __future__ import annotations
import logging  # isort:skip
import json
import os
from os.path import (
from pathlib import PurePath
from typing import TypedDict
from sphinx.errors import SphinxError
from sphinx.util import ensuredir
from sphinx.util.display import status_iterator
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import GALLERY_DETAIL, GALLERY_PAGE
from .util import _REPO_TOP
class BokehGalleryDirective(BokehDirective):
    has_content = True
    required_arguments = 0

    def run(self):
        docdir = dirname(self.env.doc2path(self.env.docname))
        gallery_file = join(docdir, 'gallery.json')
        gallery_dir = join(dirname(dirname(gallery_file)), 'gallery')
        if not exists(gallery_dir) and isdir(gallery_dir):
            raise SphinxError(f'gallery dir {gallery_dir!r} missing for gallery file {gallery_file!r}')
        gallery_json = json.load(open(gallery_file))
        opts = []
        for location in self.content:
            for detail in gallery_json[location]:
                path = PurePath('examples') / location / detail['name']
                if 'url' in detail:
                    url = detail.get('url')
                    target = '_blank'
                else:
                    url = str(path.with_suffix('.html'))
                    target = None
                opts.append({'url': url, 'target': target, 'img': str(path.with_suffix('')), 'alt': detail.get('alt'), 'title': path.stem, 'desc': detail.get('desc', None)})
        rst_text = GALLERY_PAGE.render(opts=opts)
        return self.parse(rst_text, '<bokeh-gallery>')