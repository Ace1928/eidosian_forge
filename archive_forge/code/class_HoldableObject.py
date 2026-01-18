from __future__ import annotations
from dataclasses import dataclass
import os
import abc
import typing as T
class HoldableObject(metaclass=abc.ABCMeta):
    """ Dummy base class for all objects that can be
        held by an interpreter.baseobjects.ObjectHolder """