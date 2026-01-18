from . import errors, trace, ui, urlutils
from .bzr.remote import RemoteBzrDir
from .controldir import ControlDir, format_registry
from .i18n import gettext
def _smart_upgrade_one(control_dir, format, clean_up=False, dry_run=False):
    """Convert a control directory to a new format intelligently.

    See smart_upgrade for parameter details.
    """
    dependents = None
    try:
        repo = control_dir.open_repository()
    except errors.NoRepositoryPresent:
        pass
    else:
        if repo.is_shared():
            dependents = list(repo.find_branches(using=True))
    attempted = [control_dir]
    succeeded, exceptions = _convert_items([control_dir], format, clean_up, dry_run)
    if succeeded and dependents:
        ui.ui_factory.note(gettext('Found %d dependent branches - upgrading ...') % (len(dependents),))
        branch_cdirs = [b.controldir for b in dependents]
        successes, problems = _convert_items(branch_cdirs, format, clean_up, dry_run, label='branch')
        attempted.extend(branch_cdirs)
        succeeded.extend(successes)
        exceptions.extend(problems)
    return (attempted, succeeded, exceptions)