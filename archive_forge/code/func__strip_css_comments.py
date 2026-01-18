import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
def _strip_css_comments(self, text):
    return self._CSS_COMMENTS('', text)