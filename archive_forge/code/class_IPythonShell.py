from pecan.commands import BaseCommand
from warnings import warn
import sys
class IPythonShell(object):
    """
    Open an interactive ipython shell with the Pecan app loaded.
    """

    @classmethod
    def invoke(cls, ns, banner):
        """
        :param ns: local namespace
        :param banner: interactive shell startup banner

        Embed an interactive ipython shell.
        Try the InteractiveShellEmbed API first, fall back on
        IPShellEmbed for older IPython versions.
        """
        try:
            from IPython.frontend.terminal.embed import InteractiveShellEmbed
            from IPython.frontend.terminal.ipapp import load_default_config
            config = load_default_config()
            shell = InteractiveShellEmbed(config=config, banner2=banner)
            shell(local_ns=ns)
        except ImportError:
            from IPython.Shell import IPShellEmbed
            shell = IPShellEmbed(argv=[])
            shell.set_banner(shell.IP.BANNER + '\n\n' + banner)
            shell(local_ns=ns, global_ns={})