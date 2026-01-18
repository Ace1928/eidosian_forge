from rpy2 import rinterface
from warnings import warn
from collections import defaultdict
def _map_symbols(rnames, translation=dict(), symbol_r2python=default_symbol_r2python, symbol_resolve=default_symbol_resolve):
    """
    :param names: an iterable of rnames
    :param translation: a mapping for R name->python name
    :param symbol_r2python: a function to translate an R symbol into a
                            (presumably valid) Python symbol
    :param symbol_resolve: a function to check a prospective set of
                           translation and resolve conflicts if needed
    """
    symbol_mapping = defaultdict(list)
    for rname in rnames:
        if rname in translation:
            rpyname = translation[rname]
        else:
            rpyname = symbol_r2python(rname)
        symbol_mapping[rpyname].append(rname)
    conflicts, resolutions = symbol_resolve(symbol_mapping)
    return (symbol_mapping, conflicts, resolutions)