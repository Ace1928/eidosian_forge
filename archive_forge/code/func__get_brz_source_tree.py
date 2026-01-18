import os
import platform
import sys
import breezy
from . import bedding, controldir, errors, osutils, trace
def _get_brz_source_tree():
    """Return the WorkingTree for brz source, if any.

    If brz is not being run from its working tree, returns None.
    """
    try:
        control = controldir.ControlDir.open_containing(__file__)[0]
        return control.open_workingtree(recommend_upgrade=False)
    except (errors.NotBranchError, errors.UnknownFormatError, errors.NoWorkingTree):
        return None