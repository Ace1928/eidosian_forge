from __future__ import annotations
from . import Extension
from ..inlinepatterns import InlineProcessor
import xml.etree.ElementTree as etree
import re
from typing import Any
def _getMeta(self) -> tuple[str, str, str]:
    """ Return meta data or `config` data. """
    base_url = self.config['base_url']
    end_url = self.config['end_url']
    html_class = self.config['html_class']
    if hasattr(self.md, 'Meta'):
        if 'wiki_base_url' in self.md.Meta:
            base_url = self.md.Meta['wiki_base_url'][0]
        if 'wiki_end_url' in self.md.Meta:
            end_url = self.md.Meta['wiki_end_url'][0]
        if 'wiki_html_class' in self.md.Meta:
            html_class = self.md.Meta['wiki_html_class'][0]
    return (base_url, end_url, html_class)