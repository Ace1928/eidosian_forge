from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def _fixScriptElement(self, el):
    if not self.beExtremelyLenient or not len(el.childNodes) == 1:
        return
    c = el.firstChild()
    if isinstance(c, Text):
        prefix = ''
        oldvalue = c.value
        match = self.COMMENT.match(oldvalue)
        if match:
            prefix = match.group()
            oldvalue = oldvalue[len(prefix):]
        try:
            e = parseString('<a>%s</a>' % oldvalue).childNodes[0]
        except (ParseError, MismatchedTags):
            return
        if len(e.childNodes) != 1:
            return
        e = e.firstChild()
        if isinstance(e, (CDATASection, Comment)):
            el.childNodes = []
            if prefix:
                el.childNodes.append(Text(prefix))
            el.childNodes.append(e)