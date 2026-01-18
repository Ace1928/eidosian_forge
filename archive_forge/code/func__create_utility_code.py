from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def _create_utility_code(self, template_utility_code, template_function_name):
    type_name = type_identifier(self.typedef_cname)
    utility_code = template_utility_code.specialize(type=self.typedef_cname, TypeName=type_name)
    function_name = template_function_name % type_name
    return (utility_code, function_name)