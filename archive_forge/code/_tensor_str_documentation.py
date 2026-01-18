import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
Context manager that temporarily changes the print options.  Accepted
    arguments are same as :func:`set_printoptions`.