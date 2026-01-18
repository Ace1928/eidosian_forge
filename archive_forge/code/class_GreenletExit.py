import sys
import types
import stackless
class GreenletExit(Exception):
    pass