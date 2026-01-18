import html.entities
import re
from .sgml import *
def _shorttag_replace(self, match):
    """
        :type match: Match[str]
        :rtype: str
        """
    tag = match.group(1)
    if tag in self.elements_no_end_tag:
        return '<' + tag + ' />'
    else:
        return '<' + tag + '></' + tag + '>'