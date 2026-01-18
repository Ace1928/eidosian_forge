import inspect
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from stevedore import extension
def _simple_list(mgr):
    for name in sorted(mgr.names()):
        ext = mgr[name]
        doc = _get_docstring(ext.plugin) or '\n'
        summary = doc.splitlines()[0].strip()
        yield ('* %s -- %s' % (ext.name, summary), ext.module_name)