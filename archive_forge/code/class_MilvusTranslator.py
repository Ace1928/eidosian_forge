from typing import Tuple, Union
from langchain.chains.query_constructor.ir import (
class MilvusTranslator(Visitor):
    """Translate Milvus internal query language elements to valid filters."""
    'Subset of allowed logical operators.'
    allowed_operators = [Operator.AND, Operator.NOT, Operator.OR]
    'Subset of allowed logical comparators.'
    allowed_comparators = [Comparator.EQ, Comparator.GT, Comparator.GTE, Comparator.LT, Comparator.LTE, Comparator.IN, Comparator.LIKE]

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        value = func.value
        if isinstance(func, Comparator):
            value = COMPARATOR_TO_BER[func]
        return f'{value}'

    def visit_operation(self, operation: Operation) -> str:
        if operation.operator in UNARY_OPERATORS and len(operation.arguments) == 1:
            operator = self._format_func(operation.operator)
            return operator + '(' + operation.arguments[0].accept(self) + ')'
        elif operation.operator in UNARY_OPERATORS:
            raise ValueError(f'"{operation.operator.value}" can have only one argument in Milvus')
        else:
            args = [arg.accept(self) for arg in operation.arguments]
            operator = self._format_func(operation.operator)
            return '(' + (' ' + operator + ' ').join(args) + ')'

    def visit_comparison(self, comparison: Comparison) -> str:
        comparator = self._format_func(comparison.comparator)
        processed_value = process_value(comparison.value, comparison.comparator)
        attribute = comparison.attribute
        return '( ' + attribute + ' ' + comparator + ' ' + processed_value + ' )'

    def visit_structured_query(self, structured_query: StructuredQuery) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {'expr': structured_query.filter.accept(self)}
        return (structured_query.query, kwargs)