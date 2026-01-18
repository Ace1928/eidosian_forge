import sys
def _getaction(action):
    if not action:
        return 'default'
    if action == 'all':
        return 'always'
    for a in ('default', 'always', 'ignore', 'module', 'once', 'error'):
        if a.startswith(action):
            return a
    raise _OptionError('invalid action: %r' % (action,))