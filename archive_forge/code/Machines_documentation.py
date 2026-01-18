from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
Make this an accepting state with the given action. If
        there is already an action, choose the action with highest
        priority.