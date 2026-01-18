from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _format_name_email(self, section, name, email):
    """Format name & email as a string."""
    name = self._utf8_decode('%s name' % section, name)
    email = self._utf8_decode('%s email' % section, email)
    if email:
        return '{} <{}>'.format(name, email)
    else:
        return name