import pickle
import random
from netaddr import (
class eui64_custom(eui64_unix):
    word_fmt = '%.2X'