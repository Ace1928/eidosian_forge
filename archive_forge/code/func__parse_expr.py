import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
@classmethod
def _parse_expr(cls, expr, template, lineno=-1, offset=-1):
    """Parses the given expression, raising a useful error message when a
        syntax error is encountered.
        """
    try:
        return expr and Expression(expr, template.filepath, lineno, lookup=template.lookup) or None
    except SyntaxError as err:
        err.msg += ' in expression "%s" of "%s" directive' % (expr, cls.tagname)
        raise TemplateSyntaxError(err, template.filepath, lineno, offset + (err.offset or 0))