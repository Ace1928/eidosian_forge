import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
Deserialize a treespec from a JSON string.