from collections import OrderedDict, defaultdict
from itertools import chain
from chempy.kinetics.ode import get_odesys
from chempy.units import to_unitless, linspace, logspace_from_lin

    Parameters
    ----------
    rsys : ReactionSystem
    tend : float like
    c0 : dict
        Initial concentrations.
    parameters : dict
        Parameter values.
    fig_kwargs : dict
        Keyword-arguments passed to bokeh's ``Figure``.
    slider_kwargs : dict
        Keyword-arguments passed to bokeh's ``Slider``.
    conc_bounds : dict of dicts
        Mapping substance key to dict of bounds ('start', 'end', 'step').
    x_axis_type : str
    y_axis_type : str
    integrate_kwargs : dict
        Keyword-arguments passed to integrate.
    odesys_extra : tuple
        If odesys & extra have already been generated (avoids call to ``get_odesys``).
    get_odesys_kw : dict
        Keyword-arguments passed to ``get_odesys``.
    integrate : callback
        Defaults to ``odesys.integrate``.

    