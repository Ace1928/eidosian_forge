import logging
from blazarclient import command
class DeleteFloatingIP(command.DeleteCommand):
    """Delete a floating IP."""
    resource = 'floatingip'
    allow_names = False
    log = logging.getLogger(__name__ + '.DeleteFloatingIP')