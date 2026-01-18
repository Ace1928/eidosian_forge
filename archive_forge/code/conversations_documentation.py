import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging

        Write Conversations to file from an act list.

        Conversations assume the act list is of the following form: a list of episodes,
        each of which is comprised of a list of act pairs (i.e. a list dictionaries
        returned from one parley)
        