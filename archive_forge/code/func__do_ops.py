from oslo_log import log as logging
from heat.engine import resources
def _do_ops(cinstances, opname, cnxt, stack, current_stack=None, action=None, is_stack_failure=None):
    success_count = 0
    failure = False
    failure_exception_message = None
    for ci in cinstances:
        op = getattr(ci, opname, None)
        if callable(op):
            try:
                if is_stack_failure is not None:
                    op(cnxt, stack, current_stack, action, is_stack_failure)
                else:
                    op(cnxt, stack, current_stack, action)
                success_count += 1
            except Exception as ex:
                LOG.exception('%(opname)s %(ci)s failed for %(a)s on %(sid)s', {'opname': opname, 'ci': type(ci), 'a': action, 'sid': stack.id})
                failure = True
                failure_exception_message = ex.args[0] if ex.args else str(ex)
                break
        LOG.info('done with class=%(c)s, stackid=%(sid)s, action=%(a)s', {'c': type(ci), 'sid': stack.id, 'a': action})
    return (failure, failure_exception_message, success_count)