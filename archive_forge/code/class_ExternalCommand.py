import os
from .commands import Command
class ExternalCommand(Command):
    """Class to wrap external commands."""

    @classmethod
    def find_command(cls, cmd):
        import os.path
        bzrpath = os.environ.get('BZRPATH', '')
        for dir in bzrpath.split(os.pathsep):
            if not dir:
                continue
            path = os.path.join(dir, cmd)
            if os.path.isfile(path):
                return ExternalCommand(path)
        return None

    def __init__(self, path):
        self.path = path

    def name(self):
        return os.path.basename(self.path)

    def run(self, *args, **kwargs):
        raise NotImplementedError('should not be called on %r' % self)

    def run_argv_aliases(self, argv, alias_argv=None):
        return os.spawnv(os.P_WAIT, self.path, [self.path] + argv)

    def help(self):
        m = 'external command from %s\n\n' % self.path
        pipe = os.popen('%s --help' % self.path)
        return m + pipe.read()