from . import errors, log
def _filter_revs(graph, revs, revid_range):
    if revid_range is None or revs is None:
        return revs
    return [rev for rev in revs if graph.is_between(rev[1], revid_range[0], revid_range[1])]