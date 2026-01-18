import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
class Agent(namedtuple('Agent', 'url, username')):
    """ Represents an agent that can be used for agent authentication.
    @param url(string) holds the URL of the discharger that knows about
    the agent.
    @param username holds the username agent (string).
    """