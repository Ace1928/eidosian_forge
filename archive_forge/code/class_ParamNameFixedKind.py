from inspect import Parameter
from parso import tree
from jedi.inference.utils import to_list
from jedi.inference.names import ParamNameWrapper
from jedi.inference.helpers import is_big_annoying_library
class ParamNameFixedKind(ParamNameWrapper):

    def __init__(self, param_name, new_kind):
        super().__init__(param_name)
        self._new_kind = new_kind

    def get_kind(self):
        return self._new_kind