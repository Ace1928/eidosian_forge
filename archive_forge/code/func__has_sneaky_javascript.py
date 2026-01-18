import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def _has_sneaky_javascript(self, style):
    """
        Depending on the browser, stuff like ``e x p r e s s i o n(...)``
        can get interpreted, or ``expre/* stuff */ssion(...)``.  This
        checks for attempt to do stuff like this.

        Typically the response will be to kill the entire style; if you
        have just a bit of Javascript in the style another rule will catch
        that and remove only the Javascript from the style; this catches
        more sneaky attempts.
        """
    style = self._substitute_comments('', style)
    style = style.replace('\\', '')
    style = _substitute_whitespace('', style)
    style = style.lower()
    if _has_javascript_scheme(style):
        return True
    if 'expression(' in style:
        return True
    if '@import' in style:
        return True
    if '</noscript' in style:
        return True
    if _looks_like_tag_content(style):
        return True
    return False