import dataclasses
import io
import math
from typing import Iterable, Optional
from ortools.math_opt.python import model
@dataclasses.dataclass(frozen=True)
class ModelRanges:
    """The ranges of the absolute values of the finite non-zero values in the model.

    Each range is optional since there may be no finite non-zero values
    (e.g. empty model, empty objective, all variables unbounded, ...).

    Attributes:
      objective_terms: The linear and quadratic objective terms (not including the
        offset).
      variable_bounds: The variables' lower and upper bounds.
      linear_constraint_bounds: The linear constraints' lower and upper bounds.
      linear_constraint_coefficients: The coefficients of the variables in linear
        constraints.
    """
    objective_terms: Optional[Range]
    variable_bounds: Optional[Range]
    linear_constraint_bounds: Optional[Range]
    linear_constraint_coefficients: Optional[Range]

    def __str__(self) -> str:
        """Prints the ranges in scientific format with 2 digits (i.e.

        f'{x:.2e}').

        It returns a multi-line table list of ranges. The last line does NOT end
        with a new line.

        Returns:
          The ranges in multiline string.
        """
        buf = io.StringIO()

        def print_range(prefix: str, value: Optional[Range]) -> None:
            buf.write(prefix)
            if value is None:
                buf.write('no finite values')
                return
            buf.write(f'[{value.minimum:<9.2e}, {value.maximum:<9.2e}]')
        print_range('Objective terms           : ', self.objective_terms)
        print_range('\nVariable bounds           : ', self.variable_bounds)
        print_range('\nLinear constraints bounds : ', self.linear_constraint_bounds)
        print_range('\nLinear constraints coeffs : ', self.linear_constraint_coefficients)
        return buf.getvalue()