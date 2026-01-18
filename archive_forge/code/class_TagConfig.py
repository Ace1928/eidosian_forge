import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TagConfig(object):
    """Configuration class from elyxer.config file"""
    barred = {'under': 'u'}
    family = {'sans': 'span class="sans"', 'typewriter': 'tt'}
    flex = {'CharStyle:Code': 'span class="code"', 'CharStyle:MenuItem': 'span class="menuitem"', 'Code': 'span class="code"', 'MenuItem': 'span class="menuitem"', 'Noun': 'span class="noun"', 'Strong': 'span class="strong"'}
    group = {'layouts': ['Quotation', 'Quote']}
    layouts = {'Center': 'div', 'Chapter': 'h?', 'Date': 'h2', 'Paragraph': 'div', 'Part': 'h1', 'Quotation': 'blockquote', 'Quote': 'blockquote', 'Section': 'h?', 'Subsection': 'h?', 'Subsubsection': 'h?'}
    listitems = {'Enumerate': 'ol', 'Itemize': 'ul'}
    notes = {'Comment': '', 'Greyedout': 'span class="greyedout"', 'Note': ''}
    script = {'subscript': 'sub', 'superscript': 'sup'}
    shaped = {'italic': 'i', 'slanted': 'i', 'smallcaps': 'span class="versalitas"'}