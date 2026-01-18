from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def _todict(obj):
    return dict(((k, getattr(obj, k)) for k in dir(obj) if not k.startswith('_')))