from sys import version_info as _swig_python_version_info
import weakref
class TypeRegulationsConstraint(Constraint):
    """
    The following constraint ensures that incompatibilities and requirements
    between types are respected.

    It verifies both "hard" and "temporal" incompatibilities.
    Two nodes with hard incompatible types cannot be served by the same vehicle
    at all, while with a temporal incompatibility they can't be on the same
    route at the same time.
    The VisitTypePolicy of a node determines how visiting it impacts the type
    count on the route.

    For example, for
    - three temporally incompatible types T1 T2 and T3
    - 2 pairs of nodes a1/r1 and a2/r2 of type T1 and T2 respectively, with
        - a1 and a2 of VisitTypePolicy TYPE_ADDED_TO_VEHICLE
        - r1 and r2 of policy ADDED_TYPE_REMOVED_FROM_VEHICLE
    - 3 nodes A, UV and AR of type T3, respectively with type policies
      TYPE_ADDED_TO_VEHICLE, TYPE_ON_VEHICLE_UP_TO_VISIT and
      TYPE_SIMULTANEOUSLY_ADDED_AND_REMOVED
    the configurations
    UV --> a1 --> r1 --> a2 --> r2,   a1 --> r1 --> a2 --> r2 --> A and
    a1 --> r1 --> AR --> a2 --> r2 are acceptable, whereas the configurations
    a1 --> a2 --> r1 --> ..., or A --> a1 --> r1 --> ..., or
    a1 --> r1 --> UV --> ... are not feasible.

    It also verifies same-vehicle and temporal type requirements.
    A node of type T_d with a same-vehicle requirement for type T_r needs to be
    served by the same vehicle as a node of type T_r.
    Temporal requirements, on the other hand, can take effect either when the
    dependent type is being added to the route or when it's removed from it,
    which is determined by the dependent node's VisitTypePolicy.
    In the above example:
    - If T3 is required on the same vehicle as T1, A, AR or UV must be on the
      same vehicle as a1.
    - If T2 is required when adding T1, a2 must be visited *before* a1, and if
      r2 is also visited on the route, it must be *after* a1, i.e. T2 must be on
      the vehicle when a1 is visited:
      ... --> a2 --> ... --> a1 --> ... --> r2 --> ...
    - If T3 is required when removing T1, T3 needs to be on the vehicle when
      r1 is visited:
      ... --> A --> ... --> r1 --> ...   OR   ... --> r1 --> ... --> UV --> ...
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, model):
        _pywrapcp.TypeRegulationsConstraint_swiginit(self, _pywrapcp.new_TypeRegulationsConstraint(model))

    def Post(self):
        return _pywrapcp.TypeRegulationsConstraint_Post(self)

    def InitialPropagateWrapper(self):
        return _pywrapcp.TypeRegulationsConstraint_InitialPropagateWrapper(self)
    __swig_destroy__ = _pywrapcp.delete_TypeRegulationsConstraint