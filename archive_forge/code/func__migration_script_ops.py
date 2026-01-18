from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
@_ec_dispatcher.dispatch_for(ops.MigrationScript)
def _migration_script_ops(context, directive, phase):
    """Generate a new ops.MigrationScript() for a given phase.

    E.g. given an ops.MigrationScript() directive from a vanilla autogenerate
    and an expand/contract phase name, produce a new ops.MigrationScript()
    which contains only those sub-directives appropriate to "expand" or
    "contract".  Also ensure that the branch directory exists and that
    the correct branch labels/depends_on/head revision are set up.
    """
    autogen_kwargs = {}
    version_path = upgrades.get_version_branch_path(release=upgrades.CURRENT_RELEASE, branch=phase)
    upgrades.check_bootstrap_new_branch(phase, version_path, autogen_kwargs)
    op = ops.MigrationScript(new_rev_id(), ops.UpgradeOps(ops=list(_assign_directives(context, directive.upgrade_ops.ops, phase))), ops.DowngradeOps(ops=[]), message=directive.message, **autogen_kwargs)
    if not op.upgrade_ops.is_empty():
        return op