import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once

        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        