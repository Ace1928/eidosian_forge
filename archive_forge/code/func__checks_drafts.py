import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
def _checks_drafts(both=None, draft3=None, draft4=None, raises=()):
    draft3 = draft3 or both
    draft4 = draft4 or both

    def wrap(func):
        if draft3:
            _draft_checkers['draft3'].append(draft3)
            func = FormatChecker.cls_checks(draft3, raises)(func)
        if draft4:
            _draft_checkers['draft4'].append(draft4)
            func = FormatChecker.cls_checks(draft4, raises)(func)
        return func
    return wrap