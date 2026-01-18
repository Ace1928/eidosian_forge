import functools
from debugpy.common import json, log, messaging, util
class ComponentNotAvailable(Exception):

    def __init__(self, type):
        super().__init__(f'{type.__name__} is not available')