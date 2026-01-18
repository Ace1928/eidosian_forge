from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
import oslo_i18n
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_refnode
from sphinx.util.nodes import nested_parse_with_titles
from oslo_config import cfg
from oslo_config import generator
def _list_table(headers, data, title='', columns=None):
    """Build a list-table directive.

    :param add: Function to add one row to output.
    :param headers: List of header values.
    :param data: Iterable of row data, yielding lists or tuples with rows.
    """
    yield ('.. list-table:: %s' % title)
    yield '   :header-rows: 1'
    if columns:
        yield ('   :widths: %s' % ','.join((str(c) for c in columns)))
    yield ''
    yield ('   - * %s' % headers[0])
    for h in headers[1:]:
        yield ('     * %s' % h)
    for row in data:
        yield ('   - * %s' % row[0])
        for r in row[1:]:
            yield ('     * %s' % r)