import warnings
from warnings import warn
import breezy
def activate_deprecation_warnings(override=True):
    """Call this function to activate deprecation warnings.

    When running in a 'final' release we suppress deprecation warnings.
    However, the test suite wants to see them. So when running selftest, we
    re-enable the deprecation warnings.

    Note: warnings that have already been issued under 'ignore' will not be
    reported after this point. The 'warnings' module has already marked them as
    handled, so they don't get issued again.

    :param override: If False, only add a filter if there isn't an error filter
        already. (This slightly differs from suppress_deprecation_warnings, in
        because it always overrides everything but -Werror).

    :return: A callable to remove the new warnings this added.
    """
    if not override and _check_for_filter(error_only=True):
        filter = None
    else:
        warnings.filterwarnings('default', category=DeprecationWarning)
        filter = warnings.filters[0]
    return _remove_filter_callable(filter)