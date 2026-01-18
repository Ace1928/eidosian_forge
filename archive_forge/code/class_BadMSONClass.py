from __future__ import annotations
import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
import numpy as np
import pandas as pd
import pytest
import torch
from bson.objectid import ObjectId
from monty.json import MontyDecoder, MontyEncoder, MSONable, _load_redirect, jsanitize
from . import __version__ as tests_version
class BadMSONClass(MSONable):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def as_dict(self):
        d = {'init': {'a': self.a, 'b': self.b}}
        return d