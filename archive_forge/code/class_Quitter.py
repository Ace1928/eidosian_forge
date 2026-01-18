import sys
class Quitter(object):

    def __init__(self, name, eof):
        self.name = name
        self.eof = eof

    def __repr__(self):
        return 'Use %s() or %s to exit' % (self.name, self.eof)

    def __call__(self, code=None):
        try:
            sys.stdin.close()
        except:
            pass
        raise SystemExit(code)