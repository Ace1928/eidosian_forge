from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def _exec_code_in_env(code, env, fn_name):
    exec(code, env)
    return env[fn_name]