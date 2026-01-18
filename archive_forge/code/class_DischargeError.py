import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
class DischargeError(Exception):
    """This is thrown by Client when a third party has refused a discharge"""

    def __init__(self, msg):
        super(DischargeError, self).__init__('third party refused dischargex: {}'.format(msg))