from . import errors, trace, ui, urlutils
from .bzr.remote import RemoteBzrDir
from .controldir import ControlDir, format_registry
from .i18n import gettext
def _get_object_and_label(control_dir):
    """Return the primary object and type label for a control directory.

    :return: object, label where:
      * object is a Branch, Repository or WorkingTree and
      * label is one of:
        * branch            - a branch
        * repository        - a repository
        * tree              - a lightweight checkout
    """
    try:
        try:
            br = control_dir.open_branch(unsupported=True, ignore_fallbacks=True)
        except NotImplementedError:
            br = control_dir.open_branch(ignore_fallbacks=True)
    except errors.NotBranchError:
        pass
    else:
        return (br, 'branch')
    try:
        repo = control_dir.open_repository()
    except errors.NoRepositoryPresent:
        pass
    else:
        return (repo, 'repository')
    try:
        wt = control_dir.open_workingtree()
    except (errors.NoWorkingTree, errors.NotLocalUrl):
        pass
    else:
        return (wt, 'tree')
    raise AssertionError('unknown type of control directory %s', control_dir)