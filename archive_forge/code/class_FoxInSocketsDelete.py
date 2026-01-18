from neutronclient._i18n import _
from neutronclient.common import extension
class FoxInSocketsDelete(extension.ClientExtensionDelete, FoxInSocket):
    """Delete a fox socket."""
    shell_command = 'fox-sockets-delete'