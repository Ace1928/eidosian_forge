import base64
from collections import namedtuple
import requests
from ._error import InteractionError
from ._interactor import (
from macaroonbakery._utils import visit_page_with_browser
from six.moves.urllib.parse import urljoin
class WebBrowserInteractionInfo(namedtuple('WebBrowserInteractionInfo', 'visit_url, wait_token_url')):
    """ holds the information expected in the browser-window interaction
    entry in an interaction-required error.

    :param visit_url holds the URL to be visited in a web browser.
    :param wait_token_url holds a URL that will block on GET until the browser
    interaction has completed.
    """

    @classmethod
    def from_dict(cls, info_dict):
        """Create a new instance of WebBrowserInteractionInfo, as expected
        by the Error.interaction_method method.
        @param info_dict The deserialized JSON object
        @return a new WebBrowserInteractionInfo object.
        """
        return WebBrowserInteractionInfo(visit_url=info_dict.get('VisitURL'), wait_token_url=info_dict.get('WaitTokenURL'))