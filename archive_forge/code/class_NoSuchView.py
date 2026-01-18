import re
from . import errors, osutils, transport
class NoSuchView(errors.BzrError):
    """A view does not exist.
    """
    _fmt = 'No such view: %(view_name)s.'

    def __init__(self, view_name):
        self.view_name = view_name