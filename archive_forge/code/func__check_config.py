import collections
import pickle
import pytest
import networkx as nx
from networkx.utils.configs import Config
def _check_config(self, key, value):
    if key == 'x' and value <= 0:
        raise ValueError('x must be positive')
    if key == 'y' and (not isinstance(value, str)):
        raise TypeError('y must be a str')