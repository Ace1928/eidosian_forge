from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def a1pt_theory(self, otherpoint, outframe, interframe):
    """Sets the acceleration of this point with the 1-point theory.

        The 1-point theory for point acceleration looks like this:

        ^N a^P = ^B a^P + ^N a^O + ^N alpha^B x r^OP + ^N omega^B x (^N omega^B
        x r^OP) + 2 ^N omega^B x ^B v^P

        where O is a point fixed in B, P is a point moving in B, and B is
        rotating in frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 1-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's acceleration defined in (N)
        fixedframe : ReferenceFrame
            The intermediate frame in this calculation (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> from sympy.physics.vector import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> q2 = dynamicsymbols('q2')
        >>> qd = dynamicsymbols('q', 1)
        >>> q2d = dynamicsymbols('q2', 1)
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.set_ang_vel(N, 5 * B.y)
        >>> O = Point('O')
        >>> P = O.locatenew('P', q * B.x)
        >>> P.set_vel(B, qd * B.x + q2d * B.y)
        >>> O.set_vel(N, 0)
        >>> P.a1pt_theory(O, N, B)
        (-25*q + q'')*B.x + q2''*B.y - 10*q'*B.z

        """
    _check_frame(outframe)
    _check_frame(interframe)
    self._check_point(otherpoint)
    dist = self.pos_from(otherpoint)
    v = self.vel(interframe)
    a1 = otherpoint.acc(outframe)
    a2 = self.acc(interframe)
    omega = interframe.ang_vel_in(outframe)
    alpha = interframe.ang_acc_in(outframe)
    self.set_acc(outframe, a2 + 2 * (omega ^ v) + a1 + (alpha ^ dist) + (omega ^ (omega ^ dist)))
    return self.acc(outframe)