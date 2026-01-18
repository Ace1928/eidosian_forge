import re
from hacking import core
from neutron_lib.hacking import translation_checks
def _check_namespace_imports(failure_code, namespace, new_ns, logical_line, message_override=None):
    if message_override is not None:
        msg_o = '%s: %s' % (failure_code, message_override)
    else:
        msg_o = None
    if _check_imports(namespace_imports_from_dot, namespace, logical_line):
        msg = "%s: '%s' must be used instead of '%s'." % (failure_code, logical_line.replace('%s.' % namespace, new_ns), logical_line)
        return (0, msg_o or msg)
    elif _check_imports(namespace_imports_from_root, namespace, logical_line):
        msg = "%s: '%s' must be used instead of '%s'." % (failure_code, logical_line.replace('from %s import ' % namespace, 'import %s' % new_ns), logical_line)
        return (0, msg_o or msg)
    elif _check_imports(namespace_imports_dot, namespace, logical_line):
        msg = "%s: '%s' must be used instead of '%s'." % (failure_code, logical_line.replace('import', 'from').replace('.', ' import '), logical_line)
        return (0, msg_o or msg)