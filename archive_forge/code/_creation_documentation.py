import collections.abc
import math
import warnings
from typing import cast, List, Optional, Tuple, Union
import torch

        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high)
        if required.
        