from collections import namedtuple
import torch
from . import _casting_dicts as _cd
If either of inputs is a python scalar, type-promote with NEP 50.