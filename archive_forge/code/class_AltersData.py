import functools
from collections import namedtuple
class AltersData:
    """
    Make subclasses preserve the alters_data attribute on overridden methods.
    """

    def __init_subclass__(cls, **kwargs):
        for fn_name, fn in vars(cls).items():
            if callable(fn) and (not hasattr(fn, 'alters_data')):
                for base in cls.__bases__:
                    if (base_fn := getattr(base, fn_name, None)):
                        if hasattr(base_fn, 'alters_data'):
                            fn.alters_data = base_fn.alters_data
                        break
        super().__init_subclass__(**kwargs)