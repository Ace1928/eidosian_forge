from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def a2pt_theory(self, otherpoint, outframe, fixedframe):
    """Sets the acceleration of this point with the 2-point theory.

        The 2-point theory for point acceleration looks like this:

        ^N a^P = ^N a^O + ^N alpha^B x r^OP + ^N omega^B x (^N omega^B x r^OP)

        where O and P are both points fixed in frame B, which is rotating in
        frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 2-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's acceleration defined in (N)
        fixedframe : ReferenceFrame
            The frame in which both points are fixed (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> N = ReferenceFrame('N')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> O = Point('O')
        >>> P = O.locatenew('P', 10 * B.x)
        >>> O.set_vel(N, 5 * N.x)
        >>> P.a2pt_theory(O, N, B)
        - 10*q'**2*B.x + 10*q''*B.y

        """
    _check_frame(outframe)
    _check_frame(fixedframe)
    self._check_point(otherpoint)
    dist = self.pos_from(otherpoint)
    a = otherpoint.acc(outframe)
    omega = fixedframe.ang_vel_in(outframe)
    alpha = fixedframe.ang_acc_in(outframe)
    self.set_acc(outframe, a + (alpha ^ dist) + (omega ^ (omega ^ dist)))
    return self.acc(outframe)