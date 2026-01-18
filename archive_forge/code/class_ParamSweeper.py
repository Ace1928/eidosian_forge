from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
class ParamSweeper(TemporarySubsystemManager):
    """This class enables setting values of variables/parameters
    according to a provided sequence. Iterating over this object
    sets values to the next in the sequence, at which point a
    calculation may be performed and output values compared.
    On exit, original values are restored.

    This is useful for testing a solve that is meant to perform some
    calculation, over a range of values for which the calculation
    is valid. For example:

    >>> model = ... # Make model somehow
    >>> solver = ... # Make solver somehow
    >>> input_vars = [model.v1]
    >>> n_scen = 2
    >>> input_values = ComponentMap([(model.v1, [1.1, 2.1])])
    >>> output_values = ComponentMap([(model.v2, [1.2, 2.2])])
    >>> with ParamSweeper(
    ...         n_scen,
    ...         input_values,
    ...         output_values,
    ...         to_fix=input_vars,
    ...         ) as param_sweeper:
    >>>     for inputs, outputs in param_sweeper:
    >>>         solver.solve(model)
    >>>         # inputs and outputs contain the correct values for this
    >>>         # instance of the model
    >>>         for var, val in outputs.items():
    >>>             # Test that model.v2 was calculated properly.
    >>>             # First that it equals 1.2, then that it equals 2.2
    >>>             assert var.value == val

    """

    def __init__(self, n_scenario, input_values, output_values=None, to_fix=None, to_deactivate=None, to_reset=None):
        """
        Parameters
        ----------
        n_scenario: Integer
            The number of different values we expect for each input variable
        input_values: ComponentMap
            Maps each input variable to a list of values of length n_scenario
        output_values: ComponentMap
            Maps each output variable to a list of values of length n_scenario
        to_fix: List
            to_fix argument for base class
        to_deactivate: List
            to_deactivate argument for base class
        to_reset: List
            to_reset argument for base class. This list is extended with
            input variables.

        """
        self.input_values = input_values
        output = ComponentMap() if output_values is None else output_values
        self.output_values = output
        self.n_scenario = n_scenario
        self.initial_state_values = None
        self._ip = -1
        if to_reset is None:
            to_reset = list(input_values) + list(output)
        else:
            to_reset.extend((var for var in input_values))
            to_reset.extend((var for var in output))
        super(ParamSweeper, self).__init__(to_fix=to_fix, to_deactivate=to_deactivate, to_reset=to_reset)

    def __iter__(self):
        return self

    def __next__(self):
        self._ip += 1
        i = self._ip
        n_scenario = self.n_scenario
        input_values = self.input_values
        output_values = self.output_values
        if i >= n_scenario:
            self._ip = -1
            raise StopIteration()
        else:
            inputs = ComponentMap()
            for var, values in input_values.items():
                val = values[i]
                var.set_value(val)
                inputs[var] = val
            outputs = ComponentMap([(var, values[i]) for var, values in output_values.items()])
            return (inputs, outputs)