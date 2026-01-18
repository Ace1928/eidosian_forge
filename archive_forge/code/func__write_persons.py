from __future__ import unicode_literals
import codecs
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import scan_bibtex_string
from pybtex.database.output import BaseWriter
def _write_persons(self, stream, persons, role):
    if persons:
        names = u' and '.join((self._format_name(stream, person) for person in persons))
        self._write_field(stream, role, names)