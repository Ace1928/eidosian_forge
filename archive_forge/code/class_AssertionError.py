import sys
import py
from py._code._assertionnew import interpret as reinterpret
class AssertionError(BuiltinAssertionError):

    def __init__(self, *args):
        BuiltinAssertionError.__init__(self, *args)
        if args:
            try:
                self.msg = str(args[0])
            except py.builtin._sysex:
                raise
            except:
                self.msg = '<[broken __repr__] %s at %0xd>' % (args[0].__class__, id(args[0]))
        else:
            f = py.code.Frame(sys._getframe(1))
            try:
                source = f.code.fullsource
                if source is not None:
                    try:
                        source = source.getstatement(f.lineno, assertion=True)
                    except IndexError:
                        source = None
                    else:
                        source = str(source.deindent()).strip()
            except py.error.ENOENT:
                source = None
            if source:
                self.msg = reinterpret(source, f, should_fail=True)
            else:
                self.msg = '<could not determine information>'
            if not self.args:
                self.args = (self.msg,)