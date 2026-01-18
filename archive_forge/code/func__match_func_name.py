import datetime
import warnings
from typing import Any, Literal, Optional, Sequence, Union
from langchain_core.utils import check_package_version
from typing_extensions import TypedDict
from langchain.chains.query_constructor.ir import (
def _match_func_name(self, func_name: str) -> Union[Operator, Comparator]:
    if func_name in set(Comparator):
        if self.allowed_comparators is not None:
            if func_name not in self.allowed_comparators:
                raise ValueError(f'Received disallowed comparator {func_name}. Allowed comparators are {self.allowed_comparators}')
        return Comparator(func_name)
    elif func_name in set(Operator):
        if self.allowed_operators is not None:
            if func_name not in self.allowed_operators:
                raise ValueError(f'Received disallowed operator {func_name}. Allowed operators are {self.allowed_operators}')
        return Operator(func_name)
    else:
        raise ValueError(f'Received unrecognized function {func_name}. Valid functions are {list(Operator) + list(Comparator)}')