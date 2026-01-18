import operator
from numba.core import config, utils
from numba.core.targetconfig import TargetConfig, Option
class TargetOptions:
    """Target options maps user options from decorators to the
    ``numba.core.compiler.Flags`` used by lowering and target context.
    """

    class Mapping:

        def __init__(self, flag_name, apply=lambda x: x):
            self.flag_name = flag_name
            self.apply = apply

    def finalize(self, flags, options):
        """Subclasses can override this method to make target specific
        customizations of default flags.

        Parameters
        ----------
        flags : Flags
        options : dict
        """
        pass

    @classmethod
    def parse_as_flags(cls, flags, options):
        """Parse target options defined in ``options`` and set ``flags``
        accordingly.

        Parameters
        ----------
        flags : Flags
        options : dict
        """
        opt = cls()
        opt._apply(flags, options)
        opt.finalize(flags, options)
        return flags

    def _apply(self, flags, options):
        mappings = {}
        cls = type(self)
        for k in dir(cls):
            v = getattr(cls, k)
            if isinstance(v, cls.Mapping):
                mappings[k] = v
        used = set()
        for k, mapping in mappings.items():
            if k in options:
                v = mapping.apply(options[k])
                setattr(flags, mapping.flag_name, v)
                used.add(k)
        unused = set(options) - used
        if unused:
            m = f'Unrecognized options: {unused}. Known options are {mappings.keys()}'
            raise KeyError(m)