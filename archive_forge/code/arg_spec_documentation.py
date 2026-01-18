from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.common.parameters import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.warnings import deprecate, warn
from ansible.module_utils.common.validation import (
from ansible.module_utils.errors import (
Validate ``parameters`` against argument spec.

        Error messages in the :class:`ValidationResult` may contain no_log values and should be
        sanitized with :func:`~ansible.module_utils.common.parameters.sanitize_keys` before logging or displaying.

        :arg parameters: Parameters to validate against the argument spec
        :type parameters: dict[str, dict]

        :return: :class:`ValidationResult` containing validated parameters.

        :Simple Example:

            .. code-block:: text

                argument_spec = {
                    'name': {'type': 'str'},
                    'age': {'type': 'int'},
                }

                parameters = {
                    'name': 'bo',
                    'age': '42',
                }

                validator = ArgumentSpecValidator(argument_spec)
                result = validator.validate(parameters)

                if result.error_messages:
                    sys.exit("Validation failed: {0}".format(", ".join(result.error_messages))

                valid_params = result.validated_parameters
        