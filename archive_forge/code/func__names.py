import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
def _names(node):
    if isinstance(node, _ast.Tuple):
        return tuple([_names(child) for child in node.elts])
    elif isinstance(node, _ast.Name):
        return node.id