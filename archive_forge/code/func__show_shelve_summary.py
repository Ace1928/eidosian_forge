import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
def _show_shelve_summary(params):
    """post_status hook to display a summary of shelves.

    :param params: StatusHookParams.
    """
    if params.specific_files:
        return
    get_shelf_manager = getattr(params.new_tree, 'get_shelf_manager', None)
    if get_shelf_manager is None:
        return
    try:
        manager = get_shelf_manager()
    except ShelvingUnsupported:
        mutter('shelving not supported by tree, not displaying shelves.')
    else:
        shelves = manager.active_shelves()
        if shelves:
            singular = '%d shelf exists. '
            plural = '%d shelves exist. '
            if len(shelves) == 1:
                fmt = singular
            else:
                fmt = plural
            params.to_file.write(fmt % len(shelves))
            params.to_file.write('See "brz shelve --list" for details.\n')