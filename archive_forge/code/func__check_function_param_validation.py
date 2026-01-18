from importlib import import_module
from inspect import signature
from numbers import Integral, Real
import pytest
from sklearn.utils._param_validation import (
def _check_function_param_validation(func, func_name, func_params, required_params, parameter_constraints):
    """Check that an informative error is raised when the value of a parameter does not
    have an appropriate type or value.
    """
    valid_required_params = {}
    for param_name in required_params:
        if parameter_constraints[param_name] == 'no_validation':
            valid_required_params[param_name] = 1
        else:
            valid_required_params[param_name] = generate_valid_param(make_constraint(parameter_constraints[param_name][0]))
    if func_params:
        validation_params = parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(func_params)
        missing_params = set(func_params) - set(validation_params)
        err_msg = f'Mismatch between _parameter_constraints and the parameters of {func_name}.\nConsider the unexpected parameters {unexpected_params} and expected but missing parameters {missing_params}\n'
        assert set(validation_params) == set(func_params), err_msg
    param_with_bad_type = type('BadType', (), {})()
    for param_name in func_params:
        constraints = parameter_constraints[param_name]
        if constraints == 'no_validation':
            continue
        if any((isinstance(constraint, Interval) and constraint.type == Integral for constraint in constraints)) and any((isinstance(constraint, Interval) and constraint.type == Real for constraint in constraints)):
            raise ValueError(f"The constraint for parameter {param_name} of {func_name} can't have a mix of intervals of Integral and Real types. Use the type RealNotInt instead of Real.")
        match = f"The '{param_name}' parameter of {func_name} must be .* Got .* instead."
        err_msg = f"{func_name} does not raise an informative error message when the parameter {param_name} does not have a valid type. If any Python type is valid, the constraint should be 'no_validation'."
        with pytest.raises(InvalidParameterError, match=match):
            func(**{**valid_required_params, param_name: param_with_bad_type})
            pytest.fail(err_msg)
        constraints = [make_constraint(constraint) for constraint in constraints]
        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue
            err_msg = f"{func_name} does not raise an informative error message when the parameter {param_name} does not have a valid value.\nConstraints should be disjoint. For instance [StrOptions({{'a_string'}}), str] is not a acceptable set of constraint because generating an invalid string for the first constraint will always produce a valid string for the second constraint."
            with pytest.raises(InvalidParameterError, match=match):
                func(**{**valid_required_params, param_name: bad_value})
                pytest.fail(err_msg)