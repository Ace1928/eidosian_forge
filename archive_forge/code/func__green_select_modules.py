from __future__ import annotations
import sys
import eventlet
def _green_select_modules():
    from eventlet.green import select
    modules = [('select', select)]
    from eventlet.green import selectors
    modules.append(('selectors', selectors))
    return modules