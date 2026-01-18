from torch.ao.pruning import BaseSparsifier
from functools import wraps
import warnings
import weakref
Utility that extends it to the same length as the .groups, ensuring it is a list