import inspect
import sys
from magnumclient.i18n import _
class MultipleChoices(HTTPRedirection):
    """HTTP 300 - Multiple Choices.

    Indicates multiple options for the resource that the client may follow.
    """
    http_status = 300
    message = _('Multiple Choices')