import math
from .functions import defun
class RSCache(object):

    def __init__(ctx):
        ctx._rs_cache = [0, 10, {}, {}]