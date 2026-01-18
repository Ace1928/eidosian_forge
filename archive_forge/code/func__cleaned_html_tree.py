import re
import lxml
import lxml.etree
from lxml.html.clean import Cleaner
def _cleaned_html_tree(html):
    if isinstance(html, lxml.html.HtmlElement):
        tree = html
    else:
        tree = parse_html(html)
    try:
        cleaned = cleaner.clean_html(tree)
    except AssertionError:
        cleaned = tree
    return cleaned