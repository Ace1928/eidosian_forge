from ...sage_helper import _within_sage, sage_method
from .extended_bloch import *
from ...snap import t3mlite as t3m
def _perm_for_q_tet(F, gluing):
    return _move_to_three[gluing.image(F)] * gluing * _move_from_three[F]