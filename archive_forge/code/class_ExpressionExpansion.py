from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
class ExpressionExpansion(Expansion):
    """
    Base class for expression expansions.

    https://tools.ietf.org/html/rfc6570#section-3.2
    """
    operator = ''
    partial_operator = ','
    output_prefix = ''
    var_joiner = ','
    partial_joiner = ','
    vars: list[Variable]
    trailing_joiner: str = ''

    def __init__(self, variables: str) -> None:
        super().__init__()
        if variables and variables[-1] in (',', '.', '/', ';', '&'):
            self.trailing_joiner = variables[-1]
            variables = variables[:-1]
        self.vars = [Variable(var) for var in variables.split(',')]

    @property
    def variables(self) -> Iterable[Variable]:
        """Get all variables."""
        return list(self.vars)

    @property
    def variable_names(self) -> Iterable[str]:
        """Get names of all variables."""
        return [var.name for var in self.vars]

    def _expand_var(self, variable: Variable, value: Any) -> str | None:
        """Expand a single variable."""
        return self._encode_var(variable, self._uri_encode_name(variable.name), value)

    def expand(self, values: Mapping[str, Any]) -> str | None:
        """Expand all variables, skip missing values."""
        expanded_vars: list[str] = []
        for var in self.vars:
            value = values.get(var.key, var.default)
            if value is not None:
                expanded_var = self._expand_var(var, value)
                if expanded_var is not None:
                    expanded_vars.append(expanded_var)
        if expanded_vars:
            return (self.output_prefix if not self.trailing_joiner else '') + self.var_joiner.join(expanded_vars) + self.trailing_joiner
        return None

    def partial(self, values: Mapping[str, Any]) -> str:
        """Expand all variables, replace missing values with expansions."""
        expanded_vars: list[str] = []
        missing_vars: list[Variable] = []
        result: list[tuple[list[str] | None, list[Variable] | None]] = []
        for var in self.vars:
            value = values.get(var.name, var.default)
            if value is not None:
                expanded_var = self._expand_var(var, value)
                if expanded_var is not None:
                    if missing_vars:
                        result.append((None, missing_vars))
                        missing_vars = []
                    expanded_vars.append(expanded_var)
            else:
                if expanded_vars:
                    result.append((expanded_vars, None))
                    expanded_vars = []
                missing_vars.append(var)
        if expanded_vars:
            result.append((expanded_vars, None))
        if missing_vars:
            result.append((None, missing_vars))
        output: str = ''
        first = True
        for index, (expanded, missing) in enumerate(result):
            last = index == len(result) - 1
            if expanded:
                output += (self.output_prefix if first and (not self.trailing_joiner) else '') + self.var_joiner.join(expanded) + self.trailing_joiner
            else:
                output += (self.output_prefix if first and (not last) else self.var_joiner if not last else '') + '{' + (self.operator if first else self.partial_operator) + ','.join([str(var) for var in cast('list[Variable]', missing)]) + (self.partial_joiner if not last else '') + '}'
            first = False
        return output

    def __str__(self) -> str:
        """Convert to string."""
        return '{' + self.operator + ','.join([str(var) for var in self.vars]) + self.trailing_joiner + '}'