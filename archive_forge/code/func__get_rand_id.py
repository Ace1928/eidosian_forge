import json
import uuid
import websocket
import time
import threading
from parlai.core.params import ParlaiParser
def _get_rand_id():
    """
    :return: The string of a random id using uuid4
    """
    return str(uuid.uuid4())