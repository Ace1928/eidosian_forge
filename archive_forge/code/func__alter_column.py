from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
@_ec_dispatcher.dispatch_for(ops.AlterColumnOp)
def _alter_column(context, directive, phase):
    is_expand = phase == 'expand'
    if is_expand and directive.modify_nullable is True:
        return directive
    elif not is_expand and directive.modify_nullable is False:
        return directive
    else:
        msg = _("Don't know if operation is an expand or contract at the moment: %s")
        raise NotImplementedError(msg % directive)