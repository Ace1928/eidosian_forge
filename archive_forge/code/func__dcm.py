from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def _dcm(self, parent, parent_orient):
    frames = self._dcm_cache.keys()
    dcm_dict_del = []
    dcm_cache_del = []
    if parent in frames:
        for frame in frames:
            if frame in self._dcm_dict:
                dcm_dict_del += [frame]
            dcm_cache_del += [frame]
        for frame in dcm_dict_del:
            del frame._dcm_dict[self]
        for frame in dcm_cache_del:
            del frame._dcm_cache[self]
        self._dcm_dict = self._dlist[0] = {}
        self._dcm_cache = {}
    else:
        visited = []
        queue = list(frames)
        cont = True
        while queue and cont:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                neighbors = node._dcm_dict.keys()
                for neighbor in neighbors:
                    if neighbor == parent:
                        warn('Loops are defined among the orientation of frames. This is likely not desired and may cause errors in your calculations.')
                        cont = False
                        break
                    queue.append(neighbor)
    self._dcm_dict.update({parent: parent_orient.T})
    parent._dcm_dict.update({self: parent_orient})
    self._dcm_cache.update({parent: parent_orient.T})
    parent._dcm_cache.update({self: parent_orient})