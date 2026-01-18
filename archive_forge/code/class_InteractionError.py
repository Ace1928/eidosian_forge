import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
class InteractionError(Exception):
    """This is thrown by Client when it fails to deal with an
    interaction-required error
    """

    def __init__(self, msg):
        super(InteractionError, self).__init__('cannot start interactive session: {}'.format(msg))