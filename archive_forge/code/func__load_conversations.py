import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging
def _load_conversations(self, datapath):
    if not os.path.isfile(datapath):
        raise RuntimeError(f'Conversations at path {datapath} not found. Double check your path.')
    conversations = []
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            conversations.append(Conversation(json.loads(line)))
    return conversations