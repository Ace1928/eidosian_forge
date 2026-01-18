from neutronclient._i18n import _
from neutronclient.common import extension
class FoxInSocketsShow(extension.ClientExtensionShow, FoxInSocket):
    """Show a fox socket."""
    shell_command = 'fox-sockets-show'