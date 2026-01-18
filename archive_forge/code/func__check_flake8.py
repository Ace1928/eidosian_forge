from breezy.errors import BzrError, DependencyNotPresent
from breezy.branch import Branch
def _check_flake8(local, master, old_revno, old_revid, future_revno, future_revid, tree_delta, future_tree):
    branch = local or master
    config = branch.get_config_stack()
    if config.get('flake8.pre_commit_check'):
        hook(config, tree_delta, future_tree)