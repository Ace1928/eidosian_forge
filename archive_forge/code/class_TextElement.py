import sys
import os
import re
import warnings
import types
import unicodedata
class TextElement(Element):
    """
    An element which directly contains text.

    Its children are all `Text` or `Inline` subclass nodes.  You can
    check whether an element's context is inline simply by checking whether
    its immediate parent is a `TextElement` instance (including subclasses).
    This is handy for nodes like `image` that can appear both inline and as
    standalone body elements.

    If passing children to `__init__()`, make sure to set `text` to
    ``''`` or some other suitable value.
    """
    child_text_separator = ''
    'Separator for child nodes, used by `astext()` method.'

    def __init__(self, rawsource='', text='', *children, **attributes):
        if text != '':
            textnode = Text(text)
            Element.__init__(self, rawsource, textnode, *children, **attributes)
        else:
            Element.__init__(self, rawsource, *children, **attributes)