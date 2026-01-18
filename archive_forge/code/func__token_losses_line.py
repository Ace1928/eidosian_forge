from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
from parlai.core.message import Message
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
def _token_losses_line(msg: Dict[str, Any], ignore_fields: List[str], space: str) -> Optional[str]:
    """
        Displays the loss associated with each token. Can be used for debugging
        generative models.

        See TorchGeneratorAgent._construct_token_losses for an example implementation.
        """
    key = 'token_losses'
    token_losses = msg.get(key, None)
    if key in ignore_fields or not token_losses:
        return None
    formatted_tl = ' | '.join([f'{tl[0]} {float('{:.4g}'.format(tl[1]))}' for tl in token_losses])
    return f'{space}[{key}]: {formatted_tl}'