from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
class TautologyDetected(Exception):
    """(internal) Prover uses it for reporting detected tautology"""
    pass