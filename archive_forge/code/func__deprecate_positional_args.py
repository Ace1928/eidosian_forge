from inspect import Parameter, signature
import functools
import warnings
from importlib import import_module
def _deprecate_positional_args(func=None, *, version=None):
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default=None
        The version when positional arguments will result in error.
    """
    if version is None:
        msg = 'Need to specify a version where signature will be changed'
        raise ValueError(msg)

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []
        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @functools.wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)
            args_msg = [f'{name}={arg}' for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])]
            args_msg = ', '.join(args_msg)
            warnings.warn(f'You are passing {args_msg} as a positional argument. Please change your invocation to use keyword arguments. From SciPy {version}, passing these as positional arguments will result in an error.', DeprecationWarning, stacklevel=2)
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)
        return inner_f
    if func is not None:
        return _inner_deprecate_positional_args(func)
    return _inner_deprecate_positional_args