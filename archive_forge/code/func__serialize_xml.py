import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _serialize_xml(write, elem, qnames, namespaces, short_empty_elements, **kwargs):
    tag = elem.tag
    text = elem.text
    if tag is Comment:
        write('<!--%s-->' % text)
    elif tag is ProcessingInstruction:
        write('<?%s?>' % text)
    else:
        tag = qnames[tag]
        if tag is None:
            if text:
                write(_escape_cdata(text))
            for e in elem:
                _serialize_xml(write, e, qnames, None, short_empty_elements=short_empty_elements)
        else:
            write('<' + tag)
            items = list(elem.items())
            if items or namespaces:
                if namespaces:
                    for v, k in sorted(namespaces.items(), key=lambda x: x[1]):
                        if k:
                            k = ':' + k
                        write(' xmlns%s="%s"' % (k, _escape_attrib(v)))
                for k, v in items:
                    if isinstance(k, QName):
                        k = k.text
                    if isinstance(v, QName):
                        v = qnames[v.text]
                    else:
                        v = _escape_attrib(v)
                    write(' %s="%s"' % (qnames[k], v))
            if text or len(elem) or (not short_empty_elements):
                write('>')
                if text:
                    write(_escape_cdata(text))
                for e in elem:
                    _serialize_xml(write, e, qnames, None, short_empty_elements=short_empty_elements)
                write('</' + tag + '>')
            else:
                write(' />')
    if elem.tail:
        write(_escape_cdata(elem.tail))