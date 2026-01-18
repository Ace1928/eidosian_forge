from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _augment_with_special_arguments(test_method, test_combinations):

    def decorated(self, **kwargs):
        """A wrapped test method that can treat some arguments in a special way."""
        original_kwargs = kwargs.copy()
        reasons_to_skip = []
        for combination in test_combinations:
            should_execute, reason = combination.should_execute_combination(original_kwargs.copy())
            if not should_execute:
                reasons_to_skip.append(' - ' + reason)
        if reasons_to_skip:
            self.skipTest('\n'.join(reasons_to_skip))
        customized_parameters = []
        for combination in test_combinations:
            customized_parameters.extend(combination.parameter_modifiers())
        customized_parameters = set(customized_parameters)

        def execute_test_method():
            requested_parameters = tf_inspect.getfullargspec(test_method).args
            for customized_parameter in customized_parameters:
                for argument, value in customized_parameter.modified_arguments(original_kwargs.copy(), requested_parameters).items():
                    if value is ParameterModifier.DO_NOT_PASS_TO_THE_TEST:
                        kwargs.pop(argument, None)
                    else:
                        kwargs[argument] = value
            omitted_arguments = set(requested_parameters).difference(set(list(kwargs.keys()) + ['self']))
            if omitted_arguments:
                raise ValueError('The test requires parameters whose arguments were not passed: {} .'.format(omitted_arguments))
            missing_arguments = set(list(kwargs.keys()) + ['self']).difference(set(requested_parameters))
            if missing_arguments:
                raise ValueError('The test does not take parameters that were passed : {} .'.format(missing_arguments))
            kwargs_to_pass = {}
            for parameter in requested_parameters:
                if parameter == 'self':
                    kwargs_to_pass[parameter] = self
                else:
                    kwargs_to_pass[parameter] = kwargs[parameter]
            test_method(**kwargs_to_pass)
        context_managers = []
        for combination in test_combinations:
            for manager in combination.context_managers(original_kwargs.copy()):
                context_managers.append(manager)
        if hasattr(contextlib, 'nested'):
            with contextlib.nested(*context_managers):
                execute_test_method()
        else:
            with contextlib.ExitStack() as context_stack:
                for manager in context_managers:
                    context_stack.enter_context(manager)
                execute_test_method()
    return decorated