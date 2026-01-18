from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def _dict_list(self, other, num):
    """Returns an inclusive list of reference frames that connect this
        reference frame to the provided reference frame.

        Parameters
        ==========
        other : ReferenceFrame
            The other reference frame to look for a connecting relationship to.
        num : integer
            ``0``, ``1``, and ``2`` will look for orientation, angular
            velocity, and angular acceleration relationships between the two
            frames, respectively.

        Returns
        =======
        list
            Inclusive list of reference frames that connect this reference
            frame to the other reference frame.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> C = ReferenceFrame('C')
        >>> D = ReferenceFrame('D')
        >>> B.orient_axis(A, A.x, 1.0)
        >>> C.orient_axis(B, B.x, 1.0)
        >>> D.orient_axis(C, C.x, 1.0)
        >>> D._dict_list(A, 0)
        [D, C, B, A]

        Raises
        ======

        ValueError
            When no path is found between the two reference frames or ``num``
            is an incorrect value.

        """
    connect_type = {0: 'orientation', 1: 'angular velocity', 2: 'angular acceleration'}
    if num not in connect_type.keys():
        raise ValueError('Valid values for num are 0, 1, or 2.')
    possible_connecting_paths = [[self]]
    oldlist = [[]]
    while possible_connecting_paths != oldlist:
        oldlist = possible_connecting_paths[:]
        for frame_list in possible_connecting_paths:
            frames_adjacent_to_last = frame_list[-1]._dlist[num].keys()
            for adjacent_frame in frames_adjacent_to_last:
                if adjacent_frame not in frame_list:
                    connecting_path = frame_list + [adjacent_frame]
                    if connecting_path not in possible_connecting_paths:
                        possible_connecting_paths.append(connecting_path)
    for connecting_path in oldlist:
        if connecting_path[-1] != other:
            possible_connecting_paths.remove(connecting_path)
    possible_connecting_paths.sort(key=len)
    if len(possible_connecting_paths) != 0:
        return possible_connecting_paths[0]
    msg = 'No connecting {} path found between {} and {}.'
    raise ValueError(msg.format(connect_type[num], self.name, other.name))