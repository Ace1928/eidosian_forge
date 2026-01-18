import warnings
from functools import wraps
from inspect import Parameter, signature
from typing import Iterable, Optional
def _deprecate_arguments(*, version: str, deprecated_args: Iterable[str], custom_message: Optional[str]=None):
    """Decorator to issue warnings when using deprecated arguments.

    TODO: could be useful to be able to set a custom error message.

    Args:
        version (`str`):
            The version when deprecated arguments will result in error.
        deprecated_args (`List[str]`):
            List of the arguments to be deprecated.
        custom_message (`str`, *optional*):
            Warning message that is raised. If not passed, a default warning message
            will be created.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)

        @wraps(f)
        def inner_f(*args, **kwargs):
            used_deprecated_args = []
            for _, parameter in zip(args, sig.parameters.values()):
                if parameter.name in deprecated_args:
                    used_deprecated_args.append(parameter.name)
            for kwarg_name, kwarg_value in kwargs.items():
                if kwarg_name in deprecated_args and kwarg_value != sig.parameters[kwarg_name].default:
                    used_deprecated_args.append(kwarg_name)
            if len(used_deprecated_args) > 0:
                message = f"Deprecated argument(s) used in '{f.__name__}': {', '.join(used_deprecated_args)}. Will not be supported from version '{version}'."
                if custom_message is not None:
                    message += '\n\n' + custom_message
                warnings.warn(message, FutureWarning)
            return f(*args, **kwargs)
        return inner_f
    return _inner_deprecate_positional_args