from __future__ import annotations
from . import Extension
class ExtraExtension(Extension):
    """ Add various extensions to Markdown class."""

    def __init__(self, **kwargs):
        """ `config` is a dumb holder which gets passed to the actual extension later. """
        self.config = kwargs

    def extendMarkdown(self, md):
        """ Register extension instances. """
        md.registerExtensions(extensions, self.config)