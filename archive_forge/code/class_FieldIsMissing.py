from __future__ import unicode_literals
import  six
from pybtex import richtext
from pybtex.exceptions import PybtexError
from pybtex.py3compat import fix_unicode_literals_in_doctest
class FieldIsMissing(PybtexError):

    def __init__(self, field_name, entry):
        self.field_name = field_name
        super(FieldIsMissing, self).__init__(u'missing {0} in {1}'.format(field_name, getattr(entry, 'key', '<unnamed>')))