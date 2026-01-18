from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
class TraverseNT(NamedTuple):
    depth: int
    item: Union['Traversable', 'Blob']
    src: Union['Traversable', None]