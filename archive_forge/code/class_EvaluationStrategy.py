import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
class EvaluationStrategy(ExplicitEnum):
    NO = 'no'
    STEPS = 'steps'
    EPOCH = 'epoch'