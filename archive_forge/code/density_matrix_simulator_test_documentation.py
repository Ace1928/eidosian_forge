import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
Swaps the 2nd qid |0> and |2> states when the 1st is |1>.