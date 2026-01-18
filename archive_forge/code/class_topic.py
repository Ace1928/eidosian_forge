import sys
import os
import re
import warnings
import types
import unicodedata
class topic(Structural, Element):
    """
    Topics are terminal, "leaf" mini-sections, like block quotes with titles,
    or textual figures.  A topic is just like a section, except that it has no
    subsections, and it doesn't have to conform to section placement rules.

    Topics are allowed wherever body elements (list, table, etc.) are allowed,
    but only at the top level of a section or document.  Topics cannot nest
    inside topics, sidebars, or body elements; you can't have a topic inside a
    table, list, block quote, etc.
    """