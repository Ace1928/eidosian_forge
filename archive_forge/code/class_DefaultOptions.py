from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
class DefaultOptions:
    autolinks = True
    bullets = '*+-'
    code_language = ''
    code_language_callback = None
    convert = None
    default_title = False
    escape_asterisks = True
    escape_underscores = True
    heading_style = UNDERLINED
    keep_inline_images_in = []
    newline_style = SPACES
    strip = None
    strong_em_symbol = ASTERISK
    sub_symbol = ''
    sup_symbol = ''
    wrap = False
    wrap_width = 80