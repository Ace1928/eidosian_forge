import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
class ToolboxSettings(StandardFormat):
    """This class is the base class for settings files."""

    def __init__(self):
        super().__init__()

    def parse(self, encoding=None, errors='strict', **kwargs):
        """
        Return the contents of toolbox settings file with a nested structure.

        :param encoding: encoding used by settings file
        :type encoding: str
        :param errors: Error handling scheme for codec. Same as ``decode()`` builtin method.
        :type errors: str
        :param kwargs: Keyword arguments passed to ``StandardFormat.fields()``
        :type kwargs: dict
        :rtype: ElementTree._ElementInterface
        """
        builder = TreeBuilder()
        for mkr, value in self.fields(encoding=encoding, errors=errors, **kwargs):
            block = mkr[0]
            if block in ('+', '-'):
                mkr = mkr[1:]
            else:
                block = None
            if block == '+':
                builder.start(mkr, {})
                builder.data(value)
            elif block == '-':
                builder.end(mkr)
            else:
                builder.start(mkr, {})
                builder.data(value)
                builder.end(mkr)
        return builder.close()