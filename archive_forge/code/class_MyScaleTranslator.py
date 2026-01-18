import re
from typing import Any, Callable, Dict, Tuple
from langchain.chains.query_constructor.ir import (
class MyScaleTranslator(Visitor):
    """Translate `MyScale` internal query language elements to valid filters."""
    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    'Subset of allowed logical operators.'
    allowed_comparators = [Comparator.EQ, Comparator.GT, Comparator.GTE, Comparator.LT, Comparator.LTE, Comparator.CONTAIN, Comparator.LIKE]
    map_dict = {Operator.AND: _DEFAULT_COMPOSER('AND'), Operator.OR: _DEFAULT_COMPOSER('OR'), Operator.NOT: _DEFAULT_COMPOSER('NOT'), Comparator.EQ: _DEFAULT_COMPOSER('='), Comparator.GT: _DEFAULT_COMPOSER('>'), Comparator.GTE: _DEFAULT_COMPOSER('>='), Comparator.LT: _DEFAULT_COMPOSER('<'), Comparator.LTE: _DEFAULT_COMPOSER('<='), Comparator.CONTAIN: _FUNCTION_COMPOSER('has'), Comparator.LIKE: _DEFAULT_COMPOSER('ILIKE')}

    def __init__(self, metadata_key: str='metadata') -> None:
        super().__init__()
        self.metadata_key = metadata_key

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        func = operation.operator
        self._validate_func(func)
        return self.map_dict[func](*args)

    def visit_comparison(self, comparison: Comparison) -> Dict:
        regex = '\\((.*?)\\)'
        matched = re.search('\\(\\w+\\)', comparison.attribute)
        if matched:
            attr = re.sub(regex, f'({self.metadata_key}.{matched.group(0)[1:-1]})', comparison.attribute)
        else:
            attr = f'{self.metadata_key}.{comparison.attribute}'
        value = comparison.value
        comp = comparison.comparator
        value = f"'{value}'" if isinstance(value, str) else value
        if isinstance(value, dict) and value.get('type') == 'date':
            attr = f'parseDateTime32BestEffort({attr})'
            value = f"parseDateTime32BestEffort('{value['date']}')"
        if comp is Comparator.LIKE:
            value = f"'%{value[1:-1]}%'"
        return self.map_dict[comp](attr, value)

    def visit_structured_query(self, structured_query: StructuredQuery) -> Tuple[str, dict]:
        print(structured_query)
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {'where_str': structured_query.filter.accept(self)}
        return (structured_query.query, kwargs)