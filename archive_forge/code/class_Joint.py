from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
class Joint(ABC):
    """Abstract base class for all specific joints.

    Explanation
    ===========

    A joint subtracts degrees of freedom from a body. This is the base class
    for all specific joints and holds all common methods acting as an interface
    for all joints. Custom joint can be created by inheriting Joint class and
    defining all abstract functions.

    The abstract methods are:

    - ``_generate_coordinates``
    - ``_generate_speeds``
    - ``_orient_frames``
    - ``_set_angular_velocity``
    - ``_set_linear_velocity``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Body
        The parent body of joint.
    child : Body
        The child body of joint.
    coordinates : iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Body
        The joint's parent body.
    child : Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Notes
    =====

    When providing a vector as the intermediate frame, a new intermediate frame
    is created which aligns its X axis with the provided vector. This is done
    with a single fixed rotation about a rotation axis. This rotation axis is
    determined by taking the cross product of the ``body.x`` axis with the
    provided vector. In the case where the provided vector is in the ``-body.x``
    direction, the rotation is done about the ``body.y`` axis.

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None, parent_point=None, child_point=None, parent_axis=None, child_axis=None, parent_interframe=None, child_interframe=None, parent_joint_pos=None, child_joint_pos=None):
        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name
        if not isinstance(parent, Body):
            raise TypeError('Parent must be an instance of Body.')
        self._parent = parent
        if not isinstance(child, Body):
            raise TypeError('Parent must be an instance of Body.')
        self._child = child
        self._coordinates = self._generate_coordinates(coordinates)
        self._speeds = self._generate_speeds(speeds)
        _validate_coordinates(self.coordinates, self.speeds)
        self._kdes = self._generate_kdes()
        self._parent_axis = self._axis(parent_axis, parent.frame)
        self._child_axis = self._axis(child_axis, child.frame)
        if parent_joint_pos is not None or child_joint_pos is not None:
            sympy_deprecation_warning('\n                The parent_joint_pos and child_joint_pos arguments for the Joint\n                classes are deprecated. Instead use parent_point and child_point.\n                ', deprecated_since_version='1.12', active_deprecations_target='deprecated-mechanics-joint-pos', stacklevel=4)
            if parent_point is None:
                parent_point = parent_joint_pos
            if child_point is None:
                child_point = child_joint_pos
        self._parent_point = self._locate_joint_pos(parent, parent_point)
        self._child_point = self._locate_joint_pos(child, child_point)
        if parent_axis is not None or child_axis is not None:
            sympy_deprecation_warning('\n                The parent_axis and child_axis arguments for the Joint classes\n                are deprecated. Instead use parent_interframe, child_interframe.\n                ', deprecated_since_version='1.12', active_deprecations_target='deprecated-mechanics-joint-axis', stacklevel=4)
            if parent_interframe is None:
                parent_interframe = parent_axis
            if child_interframe is None:
                child_interframe = child_axis
        self._parent_interframe = self._locate_joint_frame(parent, parent_interframe)
        self._child_interframe = self._locate_joint_frame(child, child_interframe)
        self._orient_frames()
        self._set_angular_velocity()
        self._set_linear_velocity()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        """Name of the joint."""
        return self._name

    @property
    def parent(self):
        """Parent body of Joint."""
        return self._parent

    @property
    def child(self):
        """Child body of Joint."""
        return self._child

    @property
    def coordinates(self):
        """Matrix of the joint's generalized coordinates."""
        return self._coordinates

    @property
    def speeds(self):
        """Matrix of the joint's generalized speeds."""
        return self._speeds

    @property
    def kdes(self):
        """Kinematical differential equations of the joint."""
        return self._kdes

    @property
    def parent_axis(self):
        """The axis of parent frame."""
        return self._parent_axis

    @property
    def child_axis(self):
        """The axis of child frame."""
        return self._child_axis

    @property
    def parent_point(self):
        """Attachment point where the joint is fixed to the parent body."""
        return self._parent_point

    @property
    def child_point(self):
        """Attachment point where the joint is fixed to the child body."""
        return self._child_point

    @property
    def parent_interframe(self):
        return self._parent_interframe

    @property
    def child_interframe(self):
        return self._child_interframe

    @abstractmethod
    def _generate_coordinates(self, coordinates):
        """Generate Matrix of the joint's generalized coordinates."""
        pass

    @abstractmethod
    def _generate_speeds(self, speeds):
        """Generate Matrix of the joint's generalized speeds."""
        pass

    @abstractmethod
    def _orient_frames(self):
        """Orient frames as per the joint."""
        pass

    @abstractmethod
    def _set_angular_velocity(self):
        """Set angular velocity of the joint related frames."""
        pass

    @abstractmethod
    def _set_linear_velocity(self):
        """Set velocity of related points to the joint."""
        pass

    @staticmethod
    def _to_vector(matrix, frame):
        """Converts a matrix to a vector in the given frame."""
        return Vector([(matrix, frame)])

    @staticmethod
    def _axis(ax, *frames):
        """Check whether an axis is fixed in one of the frames."""
        if ax is None:
            ax = frames[0].x
            return ax
        if not isinstance(ax, Vector):
            raise TypeError('Axis must be a Vector.')
        ref_frame = None
        for frame in frames:
            try:
                ax.to_matrix(frame)
            except ValueError:
                pass
            else:
                ref_frame = frame
                break
        if ref_frame is None:
            raise ValueError("Axis cannot be expressed in one of the body's frames.")
        if not ax.dt(ref_frame) == 0:
            raise ValueError('Axis cannot be time-varying when viewed from the associated body.')
        return ax

    @staticmethod
    def _choose_rotation_axis(frame, axis):
        components = axis.to_matrix(frame)
        x, y, z = (components[0], components[1], components[2])
        if x != 0:
            if y != 0:
                if z != 0:
                    return cross(axis, frame.x)
            if z != 0:
                return frame.y
            return frame.z
        else:
            if y != 0:
                return frame.x
            return frame.y

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

    def _generate_kdes(self):
        """Generate kinematical differential equations."""
        kdes = []
        t = dynamicsymbols._t
        for i in range(len(self.coordinates)):
            kdes.append(-self.coordinates[i].diff(t) + self.speeds[i])
        return Matrix(kdes)

    def _locate_joint_pos(self, body, joint_pos):
        """Returns the attachment point of a body."""
        if joint_pos is None:
            return body.masscenter
        if not isinstance(joint_pos, (Point, Vector)):
            raise TypeError('Attachment point must be a Point or Vector.')
        if isinstance(joint_pos, Vector):
            point_name = f'{self.name}_{body.name}_joint'
            joint_pos = body.masscenter.locatenew(point_name, joint_pos)
        if not joint_pos.pos_from(body.masscenter).dt(body.frame) == 0:
            raise ValueError('Attachment point must be fixed to the associated body.')
        return joint_pos

    def _locate_joint_frame(self, body, interframe):
        """Returns the attachment frame of a body."""
        if interframe is None:
            return body.frame
        if isinstance(interframe, Vector):
            interframe = Joint._create_aligned_interframe(body, interframe, frame_name=f'{self.name}_{body.name}_int_frame')
        elif not isinstance(interframe, ReferenceFrame):
            raise TypeError('Interframe must be a ReferenceFrame.')
        if not interframe.ang_vel_in(body.frame) == 0:
            raise ValueError(f'Interframe {interframe} is not fixed to body {body}.')
        body.masscenter.set_vel(interframe, 0)
        return interframe

    def _fill_coordinate_list(self, coordinates, n_coords, label='q', offset=0, number_single=False):
        """Helper method for _generate_coordinates and _generate_speeds.

        Parameters
        ==========

        coordinates : iterable
            Iterable of coordinates or speeds that have been provided.
        n_coords : Integer
            Number of coordinates that should be returned.
        label : String, optional
            Coordinate type either 'q' (coordinates) or 'u' (speeds). The
            Default is 'q'.
        offset : Integer
            Count offset when creating new dynamicsymbols. The default is 0.
        number_single : Boolean
            Boolean whether if n_coords == 1, number should still be used. The
            default is False.

        """

        def create_symbol(number):
            if n_coords == 1 and (not number_single):
                return dynamicsymbols(f'{label}_{self.name}')
            return dynamicsymbols(f'{label}{number}_{self.name}')
        name = 'generalized coordinate' if label == 'q' else 'generalized speed'
        generated_coordinates = []
        if coordinates is None:
            coordinates = []
        elif not iterable(coordinates):
            coordinates = [coordinates]
        if not (len(coordinates) == 0 or len(coordinates) == n_coords):
            raise ValueError(f'Expected {n_coords} {name}s, instead got {len(coordinates)} {name}s.')
        for i, coord in enumerate(coordinates):
            if coord is None:
                generated_coordinates.append(create_symbol(i + offset))
            elif isinstance(coord, (AppliedUndef, Derivative)):
                generated_coordinates.append(coord)
            else:
                raise TypeError(f'The {name} {coord} should have been a dynamicsymbol.')
        for i in range(len(coordinates) + offset, n_coords + offset):
            generated_coordinates.append(create_symbol(i))
        return Matrix(generated_coordinates)