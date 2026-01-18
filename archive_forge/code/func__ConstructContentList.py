import sys
import re
import os
import locale
from functools import reduce
def _ConstructContentList(xml_parts, specification, pretty, level=0):
    """ Appends the XML parts corresponding to the specification.

  Args:
    xml_parts: A list of XML parts to be appended to.
    specification:  The specification of the element.  See EasyXml docs.
    pretty: True if we want pretty printing with indents and new lines.
    level: Indentation level.
  """
    if pretty:
        indentation = '  ' * level
        new_line = '\n'
    else:
        indentation = ''
        new_line = ''
    name = specification[0]
    if not isinstance(name, str):
        raise Exception('The first item of an EasyXml specification should be a string.  Specification was ' + str(specification))
    xml_parts.append(indentation + '<' + name)
    rest = specification[1:]
    if rest and isinstance(rest[0], dict):
        for at, val in sorted(rest[0].items()):
            xml_parts.append(f' {at}="{_XmlEscape(val, attr=True)}"')
        rest = rest[1:]
    if rest:
        xml_parts.append('>')
        all_strings = reduce(lambda x, y: x and isinstance(y, str), rest, True)
        multi_line = not all_strings
        if multi_line and new_line:
            xml_parts.append(new_line)
        for child_spec in rest:
            if isinstance(child_spec, str):
                xml_parts.append(_XmlEscape(child_spec))
            else:
                _ConstructContentList(xml_parts, child_spec, pretty, level + 1)
        if multi_line and indentation:
            xml_parts.append(indentation)
        xml_parts.append(f'</{name}>{new_line}')
    else:
        xml_parts.append('/>%s' % new_line)