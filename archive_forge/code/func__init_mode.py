from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
@classmethod
def _init_mode(cls):
    """
        Initializes the plot mode based on
        the 'mode-specific parameters' above.
        Only intended to be called by
        PlotMode._register(). To use a mode without
        registering it, you can directly call
        ModeSubclass._init_mode().
        """

    def symbols_list(symbol_str):
        return [Symbol(s) for s in symbol_str]
    cls.i_vars = symbols_list(cls.i_vars)
    cls.d_vars = symbols_list(cls.d_vars)
    cls.i_var_count = len(cls.i_vars)
    cls.d_var_count = len(cls.d_vars)
    if cls.i_var_count > PlotMode._i_var_max:
        raise ValueError(var_count_error(True, False))
    if cls.d_var_count > PlotMode._d_var_max:
        raise ValueError(var_count_error(False, False))
    if len(cls.aliases) > 0:
        cls.primary_alias = cls.aliases[0]
    else:
        cls.primary_alias = cls.__name__
    di = cls.intervals
    if len(di) != cls.i_var_count:
        raise ValueError('Plot mode must provide a default interval for each i_var.')
    for i in range(cls.i_var_count):
        if len(di[i]) != 3:
            raise ValueError('length should be equal to 3')
        di[i] = PlotInterval(None, *di[i])
    cls._was_initialized = True