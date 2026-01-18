import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
class InteractionMethodNotFound(Exception):
    """This is thrown by client-side interaction methods when
    they find that a given interaction isn't supported by the
    client for a location"""
    pass