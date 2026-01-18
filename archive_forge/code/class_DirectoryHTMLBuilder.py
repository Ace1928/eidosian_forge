from os import path
from typing import Any, Dict, Optional
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.util import logging
from sphinx.util.osutil import SEP, os_path
class DirectoryHTMLBuilder(StandaloneHTMLBuilder):
    """
    A StandaloneHTMLBuilder that creates all HTML pages as "index.html" in
    a directory given by their pagename, so that generated URLs don't have
    ``.html`` in them.
    """
    name = 'dirhtml'

    def get_target_uri(self, docname: str, typ: Optional[str]=None) -> str:
        if docname == 'index':
            return ''
        if docname.endswith(SEP + 'index'):
            return docname[:-5]
        return docname + SEP

    def get_outfilename(self, pagename: str) -> str:
        if pagename == 'index' or pagename.endswith(SEP + 'index'):
            outfilename = path.join(self.outdir, os_path(pagename) + self.out_suffix)
        else:
            outfilename = path.join(self.outdir, os_path(pagename), 'index' + self.out_suffix)
        return outfilename