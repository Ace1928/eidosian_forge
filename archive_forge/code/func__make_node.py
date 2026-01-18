import pprint
import re
import typing as t
from markupsafe import Markup
from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context
def _make_node(self, singular: str, plural: t.Optional[str], context: t.Optional[str], variables: t.Dict[str, nodes.Expr], plural_expr: t.Optional[nodes.Expr], vars_referenced: bool, num_called_num: bool) -> nodes.Output:
    """Generates a useful node from the data provided."""
    newstyle = self.environment.newstyle_gettext
    node: nodes.Expr
    if not vars_referenced and (not newstyle):
        singular = singular.replace('%%', '%')
        if plural:
            plural = plural.replace('%%', '%')
    func_name = 'gettext'
    func_args: t.List[nodes.Expr] = [nodes.Const(singular)]
    if context is not None:
        func_args.insert(0, nodes.Const(context))
        func_name = f'p{func_name}'
    if plural_expr is not None:
        func_name = f'n{func_name}'
        func_args.extend((nodes.Const(plural), plural_expr))
    node = nodes.Call(nodes.Name(func_name, 'load'), func_args, [], None, None)
    if newstyle:
        for key, value in variables.items():
            if num_called_num and key == 'num':
                continue
            node.kwargs.append(nodes.Keyword(key, value))
    else:
        node = nodes.MarkSafeIfAutoescape(node)
        if variables:
            node = nodes.Mod(node, nodes.Dict([nodes.Pair(nodes.Const(key), value) for key, value in variables.items()]))
    return nodes.Output([node])