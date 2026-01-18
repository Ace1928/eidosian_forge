import re
from inspect import Parameter
from parso import ParserSyntaxError, parse
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.base import DefineGenericBaseClass, GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.gradual.type_var import TypeVar
from jedi.inference.helpers import is_string
from jedi.inference.compiled import builtin_from_name
from jedi.inference.param import get_executed_param_names
from jedi import debug
from jedi import parser_utils
def _find_type_from_comment_hint(context, node, varlist, name):
    index = None
    if varlist.type in ('testlist_star_expr', 'exprlist', 'testlist'):
        index = 0
        for child in varlist.children:
            if child == name:
                break
            if child.type == 'operator':
                continue
            index += 1
        else:
            return []
    comment = parser_utils.get_following_comment_same_line(node)
    if comment is None:
        return []
    match = re.match('^#\\s*type:\\s*([^#]*)', comment)
    if match is None:
        return []
    return _infer_annotation_string(context, match.group(1).strip(), index).execute_annotation()