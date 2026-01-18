import itertools
def _update_other_results(results, best):
    """Copied from _update_other_results in numpy/core/einsumfunc.py

    Update the positions and provisional input_sets of ``results`` based on
    performing the contraction result ``best``. Remove any involving the tensors
    contracted.

    Parameters
    ----------
    results : list
        List of contraction results produced by ``_parse_possible_contraction``.
    best : list
        The best contraction of ``results`` i.e. the one that will be performed.

    Returns
    -------
    mod_results : list
        The list of modifed results, updated with outcome of ``best`` contraction.  # NOQA
    """
    best_con = best[1]
    bx, by = best_con
    mod_results = []
    for cost, (x, y), con_sets in results:
        if x in best_con or y in best_con:
            continue
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])
        mod_con = (x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by))
        mod_results.append((cost, mod_con, con_sets))
    return mod_results