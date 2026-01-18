from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
def _assign_directives(context, directives, phase=None):
    for directive in directives:
        decider = _ec_dispatcher.dispatch(directive)
        if phase is None:
            phases = upgrades.MIGRATION_BRANCHES
        else:
            phases = (phase,)
        for phase in phases:
            decided = decider(context, directive, phase)
            if decided:
                yield decided