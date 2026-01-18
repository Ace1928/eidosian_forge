import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging
class Turn(AttrDict):
    """
    Utility class for a dialog turn.
    """

    def __init__(self, id=None, text=None, **kwargs):
        super().__init__(self, id=id, text=text, **kwargs)