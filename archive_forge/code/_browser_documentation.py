import base64
from collections import namedtuple
import requests
from ._error import InteractionError
from ._interactor import (
from macaroonbakery._utils import visit_page_with_browser
from six.moves.urllib.parse import urljoin
Create a new instance of WebBrowserInteractionInfo, as expected
        by the Error.interaction_method method.
        @param info_dict The deserialized JSON object
        @return a new WebBrowserInteractionInfo object.
        