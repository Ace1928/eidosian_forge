import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def _make_template(query):
    query_template = []
    for constraint in query:
        if isinstance(constraint, tuple):
            query_template.append((constraint[0], constraint[1], '%%(%s)s' % constraint[2]))
        elif isinstance(constraint, str):
            query_template.append(constraint)
        elif isinstance(constraint, list):
            query_template.append(_make_template(constraint))
        else:
            raise ProgrammingError('Unrecognized token in query: %s' % constraint)
    return query_template