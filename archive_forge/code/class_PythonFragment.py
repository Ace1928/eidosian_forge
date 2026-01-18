import re
from mako import exceptions
from mako import pyparser
class PythonFragment(PythonCode):
    """extends PythonCode to provide identifier lookups in partial control
    statements

    e.g.::

        for x in 5:
        elif y==9:
        except (MyException, e):

    """

    def __init__(self, code, **exception_kwargs):
        m = re.match('^(\\w+)(?:\\s+(.*?))?:\\s*(#|$)', code.strip(), re.S)
        if not m:
            raise exceptions.CompileException("Fragment '%s' is not a partial control statement" % code, **exception_kwargs)
        if m.group(3):
            code = code[:m.start(3)]
        keyword, expr = m.group(1, 2)
        if keyword in ['for', 'if', 'while']:
            code = code + 'pass'
        elif keyword == 'try':
            code = code + 'pass\nexcept:pass'
        elif keyword in ['elif', 'else']:
            code = 'if False:pass\n' + code + 'pass'
        elif keyword == 'except':
            code = 'try:pass\n' + code + 'pass'
        elif keyword == 'with':
            code = code + 'pass'
        else:
            raise exceptions.CompileException("Unsupported control keyword: '%s'" % keyword, **exception_kwargs)
        super().__init__(code, **exception_kwargs)