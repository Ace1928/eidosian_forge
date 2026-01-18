from collections import OrderedDict
from functools import reduce, partial
from itertools import chain
from operator import attrgetter, mul
import math
import warnings
from ..units import (
from ..util.pyutil import deprecated
from ..util._expr import Expr, Symbol
from .rates import RateExpr, MassAction
def _create_odesys(rsys, substance_symbols=None, parameter_symbols=None, pretty_replace=lambda x: x, backend=None, SymbolicSys=None, time_symbol=None, unit_registry=None, rates_kw=None, parameter_expressions=None, symbolic_kw=None):
    """This will be a simpler version of get_odesys without the unit handling code.
    The motivation is to reduce complexity (the code of get_odesys is long with multiple closures).

    This will also rely on SymPy explicitly and the user will be expected to deal with SymPy
    expressions.

    Only when this function has the same capabilities as get_odesys will it become and public API
    (along with a deprecation of get_odesys).

    Parameters
    ----------
    rsys : ReactionSystem instance
    substance_symbols : OrderedDict
       If ``None``: ``rsys.substances`` will be used.
    parameter_symbols : OrderedDict
    backend : str or module
        Symbolic backend (e.g. sympy). The package ``sym`` is used as a wrapper.
    SymbolicSys: class
        See ``pyodesys`` for API.
    time_symbol : Symbol
    unit_registry : object
        e.g. ``chempy.units.SI_base_registry``
    rates_kw : dict
        Keyword arguments passed to the ``rates`` method of rsys.
    parameter_expressions : dict
        Optional overrides.
    symbolic_kw : dict
        Keyword arguments passed on to SymbolicSys.

    Returns
    -------
    SymbolicSys (subclass of ``pyodesys.ODESys``)
    dict :
        - ``'symbols'``: dict mapping str to symbol.
        - ``'validate'``: callable acppeting a dictionary mapping str to quantities
    """
    if backend is None:
        from sym import Backend
        backend = Backend(backend)
    if SymbolicSys is None:
        from pyodesys.symbolic import SymbolicSys
    if substance_symbols is None:
        substance_symbols = OrderedDict([(key, backend.Symbol(key)) for key in rsys.substances])
    if isinstance(substance_symbols, OrderedDict):
        if list(substance_symbols) != list(rsys.substances):
            raise ValueError('substance_symbols needs to have same (oredered) keys as rsys.substances')
    if parameter_symbols is None:
        keys = []
        for rxnpar in map(attrgetter('param'), rsys.rxns):
            if isinstance(rxnpar, str):
                if rxnpar in (parameter_expressions or {}):
                    for pk in parameter_expressions[rxnpar].all_parameter_keys():
                        keys.append(pk)
                else:
                    keys.append(rxnpar)
            elif isinstance(rxnpar, Expr):
                keys.extend(rxnpar.all_unique_keys())
                for pk in rxnpar.all_parameter_keys():
                    if pk not in keys:
                        keys.append(pk)
            else:
                raise NotImplementedError('Unknown')
        if rates_kw and 'cstr_fr_fc' in rates_kw:
            flowrate_volume, feed_conc = rates_kw['cstr_fr_fc']
            keys.append(flowrate_volume)
            keys.extend(feed_conc.values())
            assert all((sk in rsys.substances for sk in feed_conc))
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicates in keys')
        parameter_symbols = OrderedDict([(key, backend.Symbol(key)) for key in keys])
    if not isinstance(parameter_symbols, OrderedDict):
        raise ValueError('parameter_symbols needs to be an OrderedDict')
    symbols = OrderedDict(chain(substance_symbols.items(), parameter_symbols.items()))
    symbols['time'] = time_symbol or backend.Symbol('t')
    if any((symbols['time'] == v for k, v in symbols.items() if k != 'time')):
        raise ValueError('time_symbol already in use (name clash?)')
    varbls = dict(symbols, **parameter_symbols)
    varbls.update(parameter_expressions or {})
    rates = rsys.rates(varbls, **rates_kw or {})
    compo_vecs, compo_names = rsys.composition_balance_vectors()
    odesys = SymbolicSys(zip([substance_symbols[key] for key in rsys.substances], [rates[key] for key in rsys.substances]), symbols['time'], parameter_symbols.values(), names=list(rsys.substances.keys()), latex_names=[s.latex_name for s in rsys.substances.values()], param_names=parameter_symbols.keys(), latex_param_names=[pretty_replace(n) for n in parameter_symbols.keys()], linear_invariants=compo_vecs, linear_invariant_names=list(map(str, compo_names)), backend=backend, dep_by_name=True, par_by_name=True, **symbolic_kw or {})
    validate = partial(_validate, rsys=rsys, symbols=symbols, odesys=odesys, backend=backend)
    return (odesys, {'symbols': symbols, 'validate': validate, 'unit_aware_solve': _mk_unit_aware_solve(odesys, unit_registry, validate=validate) if unit_registry else None})