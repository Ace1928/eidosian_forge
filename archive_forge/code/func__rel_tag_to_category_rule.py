import codecs
import copy
import json
import os
from urllib.parse import unquote
import bs4
from . import mf2_classes
from .dom_helpers import get_children
from .mf_helpers import unordered_list
def _rel_tag_to_category_rule(child, html_parser, **kwargs):
    """rel=tag converts to p-category using a special transformation (the
    category becomes the tag href's last path segment). This rule adds a new data tag so that
    <a rel="tag" href="http://example.com/tags/cat"></a> gets replaced with
    <data class="p-category" value="cat"></data>
    """
    href = child.get('href', '')
    rels = child.get('rel', [])
    if 'tag' in rels and href:
        segments = [seg for seg in href.split('/') if seg]
        if segments:
            if html_parser:
                soup = bs4.BeautifulSoup('', features=html_parser)
            else:
                soup = bs4.BeautifulSoup('')
            data = soup.new_tag('data')
            data['class'] = ['p-category']
            data['value'] = unquote(segments[-1])
            child.insert_before(data)
            child['rel'] = [r for r in rels if r != 'tag']