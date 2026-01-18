from itertools import chain
from genshi.core import Attrs, Markup, Namespace, Stream
from genshi.core import START, END, START_NS, END_NS, TEXT, PI, COMMENT
from genshi.input import XMLParser
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.interpolation import interpolate
from genshi.template.directives import *
from genshi.template.text import NewTextTemplate
def add_directives(self, namespace, factory):
    """Register a custom `DirectiveFactory` for a given namespace.
        
        :param namespace: the namespace URI
        :type namespace: `basestring`
        :param factory: the directive factory to register
        :type factory: `DirectiveFactory`
        :since: version 0.6
        """
    assert not self._prepared, 'Too late for adding directives, template already prepared'
    self._stream = self._extract_directives(self._stream, namespace, factory)