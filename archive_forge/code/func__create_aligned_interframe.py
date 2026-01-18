from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
@staticmethod
def _create_aligned_interframe(frame, align_axis, frame_axis=None, frame_name=None):
    """
        Returns an intermediate frame, where the ``frame_axis`` defined in
        ``frame`` is aligned with ``axis``. By default this means that the X
        axis will be aligned with ``axis``.

        Parameters
        ==========

        frame : Body or ReferenceFrame
            The body or reference frame with respect to which the intermediate
            frame is oriented.
        align_axis : Vector
            The vector with respect to which the intermediate frame will be
            aligned.
        frame_axis : Vector
            The vector of the frame which should get aligned with ``axis``. The
            default is the X axis of the frame.
        frame_name : string
            Name of the to be created intermediate frame. The default adds
            "_int_frame" to the name of ``frame``.

        Example
        =======

        An intermediate frame, where the X axis of the parent becomes aligned
        with ``parent.y + parent.z`` can be created as follows:

        >>> from sympy.physics.mechanics.joint import Joint
        >>> from sympy.physics.mechanics import Body
        >>> parent = Body('parent')
        >>> parent_interframe = Joint._create_aligned_interframe(
        ...     parent, parent.y + parent.z)
        >>> parent_interframe
        parent_int_frame
        >>> parent.dcm(parent_interframe)
        Matrix([
        [        0, -sqrt(2)/2, -sqrt(2)/2],
        [sqrt(2)/2,        1/2,       -1/2],
        [sqrt(2)/2,       -1/2,        1/2]])
        >>> (parent.y + parent.z).express(parent_interframe)
        sqrt(2)*parent_int_frame.x

        Notes
        =====

        The direction cosine matrix between the given frame and intermediate
        frame is formed using a simple rotation about an axis that is normal to
        both ``align_axis`` and ``frame_axis``. In general, the normal axis is
        formed by crossing the ``frame_axis`` with the ``align_axis``. The
        exception is if the axes are parallel with opposite directions, in which
        case the rotation vector is chosen using the rules in the following
        table with the vectors expressed in the given frame:

        .. list-table::
           :header-rows: 1

           * - ``align_axis``
             - ``frame_axis``
             - ``rotation_axis``
           * - ``-x``
             - ``x``
             - ``z``
           * - ``-y``
             - ``y``
             - ``x``
           * - ``-z``
             - ``z``
             - ``y``
           * - ``-x-y``
             - ``x+y``
             - ``z``
           * - ``-y-z``
             - ``y+z``
             - ``x``
           * - ``-x-z``
             - ``x+z``
             - ``y``
           * - ``-x-y-z``
             - ``x+y+z``
             - ``(x+y+z) × x``

        """
    if isinstance(frame, Body):
        frame = frame.frame
    if frame_axis is None:
        frame_axis = frame.x
    if frame_name is None:
        if frame.name[-6:] == '_frame':
            frame_name = f'{frame.name[:-6]}_int_frame'
        else:
            frame_name = f'{frame.name}_int_frame'
    angle = frame_axis.angle_between(align_axis)
    rotation_axis = cross(frame_axis, align_axis)
    if rotation_axis == Vector(0) and angle == 0:
        return frame
    if angle == pi:
        rotation_axis = Joint._choose_rotation_axis(frame, align_axis)
    int_frame = ReferenceFrame(frame_name)
    int_frame.orient_axis(frame, rotation_axis, angle)
    int_frame.set_ang_vel(frame, 0 * rotation_axis)
    return int_frame