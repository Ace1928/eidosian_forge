import contextlib
import functools
import inspect
import math
import warnings
def _generate_deprecation_warning(since, message='', name='', alternative='', pending=False, obj_type='', addendum='', *, removal=''):
    if pending:
        if removal:
            raise ValueError('A pending deprecation cannot have a scheduled removal')
    else:
        removal = f'in {removal}' if removal else 'two minor releases later'
    if not message:
        message = ('The %(name)s %(obj_type)s' if obj_type else '%(name)s') + (' will be deprecated in a future version' if pending else ' was deprecated in Matplotlib %(since)s' + (' and will be removed %(removal)s' if removal else '')) + '.' + (' Use %(alternative)s instead.' if alternative else '') + (' %(addendum)s' if addendum else '')
    warning_cls = PendingDeprecationWarning if pending else MatplotlibDeprecationWarning
    return warning_cls(message % dict(func=name, name=name, obj_type=obj_type, since=since, removal=removal, alternative=alternative, addendum=addendum))