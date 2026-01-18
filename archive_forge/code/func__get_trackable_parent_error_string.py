import gc
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.trackable import base as trackable
def _get_trackable_parent_error_string(capture):
    """Gets error string with the capture's parent object."""
    parent = getattr(capture, '_parent_trackable', None)
    if parent is not None:
        return f'Trackable referencing this tensor = {parent()}'
    trackable_referrers = []
    for primary_referrer in gc.get_referrers(capture):
        if isinstance(primary_referrer, trackable.Trackable):
            trackable_referrers.append(primary_referrer)
        for secondary_referrer in gc.get_referrers(primary_referrer):
            if isinstance(secondary_referrer, trackable.Trackable):
                trackable_referrers.append(secondary_referrer)
    return 'Trackable Python objects referring to this tensor (from gc.get_referrers, limited to two hops) = [\n\t\t{}]'.format('\n\t\t'.join([repr(obj) for obj in trackable_referrers]))