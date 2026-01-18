from functools import wraps
from typing import Callable, TypeVar, cast
from ..warnings import SetuptoolsDeprecationWarning
from . import setupcfg
def _deprecation_notice(fn: Fn) -> Fn:

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        SetuptoolsDeprecationWarning.emit('Deprecated API usage.', f'\n            As setuptools moves its configuration towards `pyproject.toml`,\n            `{__name__}.{fn.__name__}` became deprecated.\n\n            For the time being, you can use the `{setupcfg.__name__}` module\n            to access a backward compatible API, but this module is provisional\n            and might be removed in the future.\n\n            To read project metadata, consider using\n            ``build.util.project_wheel_metadata`` (https://pypi.org/project/build/).\n            For simple scenarios, you can also try parsing the file directly\n            with the help of ``configparser``.\n            ')
        return fn(*args, **kwargs)
    return cast(Fn, _wrapper)