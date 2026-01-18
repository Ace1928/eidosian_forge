import collections
def _get_initial_max_subplot_ids():
    max_subplot_ids = {subplot_type: 0 for subplot_type in _single_subplot_types}
    max_subplot_ids['xaxis'] = 0
    max_subplot_ids['yaxis'] = 0
    return max_subplot_ids