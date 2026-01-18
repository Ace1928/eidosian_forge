from manilaclient.common._i18n import _
from manilaclient.common.apiclient.exceptions import *  # noqa
class ManilaclientException(Exception):
    """A generic client error."""
    message = _('An unexpected error occured.')

    def __init__(self, message):
        self.message = message or self.message

    def __str__(self):
        return self.message