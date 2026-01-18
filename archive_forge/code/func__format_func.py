from typing import Dict, Tuple, Union
from langchain.chains.query_constructor.ir import (
def _format_func(self, func: Union[Operator, Comparator]) -> str:
    self._validate_func(func)
    map_dict = {Operator.AND: '$and', Operator.OR: '$or', Comparator.EQ: '$eq', Comparator.NE: '$ne', Comparator.GTE: '$gte', Comparator.LTE: '$lte', Comparator.LT: '$lt', Comparator.GT: '$gt', Comparator.IN: '$in', Comparator.NIN: '$nin'}
    return map_dict[func]