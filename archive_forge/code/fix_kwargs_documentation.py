from lib2to3 import fixer_base
from lib2to3.fixer_util import token, String, Newline, Comma, Name
from libfuturize.fixer_util import indentation, suitify, DoubleStar

    Returns string with the name of the kwargs dict if the params after the first star need fixing
    Otherwise returns empty string
    