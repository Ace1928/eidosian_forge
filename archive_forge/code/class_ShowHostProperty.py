import logging
from blazarclient import command
from blazarclient import exception
class ShowHostProperty(command.ShowPropertyCommand):
    """Show host property."""
    resource = 'host'
    json_indent = 4
    log = logging.getLogger(__name__ + '.ShowHostProperty')