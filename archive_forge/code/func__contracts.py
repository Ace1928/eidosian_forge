from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
@_ec_dispatcher.dispatch_for(ops.DropConstraintOp)
@_ec_dispatcher.dispatch_for(ops.DropIndexOp)
@_ec_dispatcher.dispatch_for(ops.DropTableOp)
@_ec_dispatcher.dispatch_for(ops.DropColumnOp)
def _contracts(context, directive, phase):
    if phase == 'contract':
        return directive
    else:
        return None