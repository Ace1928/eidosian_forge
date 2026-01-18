import re
from textwrap import dedent
from inspect import Parameter
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
def _complete_code_lines(self, code_lines):
    module_node = self._inference_state.grammar.parse(''.join(code_lines))
    module_value = DocstringModule(in_module_context=self._module_context, inference_state=self._inference_state, module_node=module_node, code_lines=code_lines)
    return Completion(self._inference_state, module_value.as_context(), code_lines=code_lines, position=module_node.end_pos, signatures_callback=lambda *args, **kwargs: [], fuzzy=self._fuzzy).complete()