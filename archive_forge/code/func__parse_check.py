import logging
import re
from oslo_policy import _checks
def _parse_check(rule):
    """Parse a single base check rule into an appropriate Check object."""
    if rule == '!':
        return _checks.FalseCheck()
    elif rule == '@':
        return _checks.TrueCheck()
    try:
        kind, match = rule.split(':', 1)
    except Exception:
        LOG.exception('Failed to understand rule %s', rule)
        return _checks.FalseCheck()
    extension_checks = _checks.get_extensions()
    if kind in extension_checks:
        return extension_checks[kind](kind, match)
    elif kind in _checks.registered_checks:
        return _checks.registered_checks[kind](kind, match)
    elif None in _checks.registered_checks:
        return _checks.registered_checks[None](kind, match)
    else:
        LOG.error('No handler for matches of kind %s', kind)
        return _checks.FalseCheck()