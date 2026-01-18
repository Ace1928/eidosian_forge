import inspect
import logging
import sys
def _set_ev_cls_dec(handler):
    if 'callers' not in dir(handler):
        handler.callers = {}
    for e in _listify(ev_cls):
        handler.callers[e] = _Caller(_listify(dispatchers), None)
    return handler