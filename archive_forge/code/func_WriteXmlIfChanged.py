import sys
import re
import os
import locale
from functools import reduce
def WriteXmlIfChanged(content, path, encoding='utf-8', pretty=False, win32=sys.platform == 'win32'):
    """ Writes the XML content to disk, touching the file only if it has changed.

  Args:
    content:  The structured content to be written.
    path: Location of the file.
    encoding: The encoding to report on the first line of the XML file.
    pretty: True if we want pretty printing with indents and new lines.
  """
    xml_string = XmlToString(content, encoding, pretty)
    if win32 and os.linesep != '\r\n':
        xml_string = xml_string.replace('\n', '\r\n')
    default_encoding = locale.getdefaultlocale()[1]
    if default_encoding and default_encoding.upper() != encoding.upper():
        xml_string = xml_string.encode(encoding)
    try:
        with open(path) as file:
            existing = file.read()
    except OSError:
        existing = None
    if existing != xml_string:
        with open(path, 'wb') as file:
            file.write(xml_string)