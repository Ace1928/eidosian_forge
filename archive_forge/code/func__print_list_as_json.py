import os
import sys
import shutil
import argparse
from textwrap import dedent
from pip._vendor.pygments import __version__, highlight
from pip._vendor.pygments.util import ClassNotFound, OptionError, docstring_headline, \
from pip._vendor.pygments.lexers import get_all_lexers, get_lexer_by_name, guess_lexer, \
from pip._vendor.pygments.lexers.special import TextLexer
from pip._vendor.pygments.formatters.latex import LatexEmbeddedLexer, LatexFormatter
from pip._vendor.pygments.formatters import get_all_formatters, get_formatter_by_name, \
from pip._vendor.pygments.formatters.terminal import TerminalFormatter
from pip._vendor.pygments.formatters.terminal256 import Terminal256Formatter, TerminalTrueColorFormatter
from pip._vendor.pygments.filters import get_all_filters, find_filter_class
from pip._vendor.pygments.styles import get_all_styles, get_style_by_name
def _print_list_as_json(requested_items):
    import json
    result = {}
    if 'lexer' in requested_items:
        info = {}
        for fullname, names, filenames, mimetypes in get_all_lexers():
            info[fullname] = {'aliases': names, 'filenames': filenames, 'mimetypes': mimetypes}
        result['lexers'] = info
    if 'formatter' in requested_items:
        info = {}
        for cls in get_all_formatters():
            doc = docstring_headline(cls)
            info[cls.name] = {'aliases': cls.aliases, 'filenames': cls.filenames, 'doc': doc}
        result['formatters'] = info
    if 'filter' in requested_items:
        info = {}
        for name in get_all_filters():
            cls = find_filter_class(name)
            info[name] = {'doc': docstring_headline(cls)}
        result['filters'] = info
    if 'style' in requested_items:
        info = {}
        for name in get_all_styles():
            cls = get_style_by_name(name)
            info[name] = {'doc': docstring_headline(cls)}
        result['styles'] = info
    json.dump(result, sys.stdout)