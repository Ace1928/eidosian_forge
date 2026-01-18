from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
@_ec_dispatcher.dispatch_for(ops.ModifyTableOps)
def _modify_table_ops(context, directive, phase):
    op = ops.ModifyTableOps(directive.table_name, ops=list(_assign_directives(context, directive.ops, phase)), schema=directive.schema)
    if not op.is_empty():
        return op