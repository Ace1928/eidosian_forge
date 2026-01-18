from pecan.commands import BaseCommand
from warnings import warn
import sys
class NativePythonShell(object):
    """
    Open an interactive python shell with the Pecan app loaded.
    """

    @classmethod
    def invoke(cls, ns, banner):
        """
        :param ns: local namespace
        :param banner: interactive shell startup banner

        Embed an interactive native python shell.
        """
        import code
        py_prefix = sys.platform.startswith('java') and 'J' or 'P'
        shell_banner = 'Pecan Interactive Shell\n%sython %s\n\n' % (py_prefix, sys.version)
        shell = code.InteractiveConsole(locals=ns)
        try:
            import readline
        except ImportError:
            pass
        shell.interact(shell_banner + banner)