import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def _sum_and_sub_mul(a, b, c):
    return ((a + b) * c, (a - b) * c)