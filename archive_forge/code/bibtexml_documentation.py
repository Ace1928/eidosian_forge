from __future__ import unicode_literals
import io
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
import six
from pybtex.database.output import BaseWriter

        Return a unicode XML string without encoding declaration.

        >>> from pybtex.database import BibliographyData
        >>> data = BibliographyData()
        >>> unicode_xml = Writer().to_string(data)
        >>> isinstance(unicode_xml, six.text_type)
        True
        >>> print(unicode_xml)
        <bibtex:file xmlns:bibtex="http://bibtexml.sf.net/">
        <BLANKLINE>
        </bibtex:file>
        