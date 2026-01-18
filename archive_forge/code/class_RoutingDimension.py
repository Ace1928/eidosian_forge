from sys import version_info as _swig_python_version_info
import weakref
class RoutingDimension(object):
    """
    Dimensions represent quantities accumulated at nodes along the routes. They
    represent quantities such as weights or volumes carried along the route, or
    distance or times.

    Quantities at a node are represented by "cumul" variables and the increase
    or decrease of quantities between nodes are represented by "transit"
    variables. These variables are linked as follows:

    if j == next(i),
    cumuls(j) = cumuls(i) + transits(i) + slacks(i) +
                state_dependent_transits(i)

    where slack is a positive slack variable (can represent waiting times for
    a time dimension), and state_dependent_transits is a non-purely functional
    version of transits_. Favour transits over state_dependent_transits when
    possible, because purely functional callbacks allow more optimisations and
    make the model faster and easier to solve.
    for a given vehicle, it is passed as an external vector, it would be better
    to have this information here.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr
    __swig_destroy__ = _pywrapcp.delete_RoutingDimension

    def model(self):
        """ Returns the model on which the dimension was created."""
        return _pywrapcp.RoutingDimension_model(self)

    def GetTransitValue(self, from_index, to_index, vehicle):
        """
        Returns the transition value for a given pair of nodes (as var index);
        this value is the one taken by the corresponding transit variable when
        the 'next' variable for 'from_index' is bound to 'to_index'.
        """
        return _pywrapcp.RoutingDimension_GetTransitValue(self, from_index, to_index, vehicle)

    def GetTransitValueFromClass(self, from_index, to_index, vehicle_class):
        """
        Same as above but taking a vehicle class of the dimension instead of a
        vehicle (the class of a vehicle can be obtained with vehicle_to_class()).
        """
        return _pywrapcp.RoutingDimension_GetTransitValueFromClass(self, from_index, to_index, vehicle_class)

    def CumulVar(self, index):
        """
        Get the cumul, transit and slack variables for the given node (given as
        int64_t var index).
        """
        return _pywrapcp.RoutingDimension_CumulVar(self, index)

    def TransitVar(self, index):
        return _pywrapcp.RoutingDimension_TransitVar(self, index)

    def FixedTransitVar(self, index):
        return _pywrapcp.RoutingDimension_FixedTransitVar(self, index)

    def SlackVar(self, index):
        return _pywrapcp.RoutingDimension_SlackVar(self, index)

    def SetSpanUpperBoundForVehicle(self, upper_bound, vehicle):
        """
        Sets an upper bound on the dimension span on a given vehicle. This is the
        preferred way to limit the "length" of the route of a vehicle according to
        a dimension.
        """
        return _pywrapcp.RoutingDimension_SetSpanUpperBoundForVehicle(self, upper_bound, vehicle)

    def SetSpanCostCoefficientForVehicle(self, coefficient, vehicle):
        """
        Sets a cost proportional to the dimension span on a given vehicle,
        or on all vehicles at once. "coefficient" must be nonnegative.
        This is handy to model costs proportional to idle time when the dimension
        represents time.
        The cost for a vehicle is
          span_cost = coefficient * (dimension end value - dimension start value).
        """
        return _pywrapcp.RoutingDimension_SetSpanCostCoefficientForVehicle(self, coefficient, vehicle)

    def SetSpanCostCoefficientForAllVehicles(self, coefficient):
        return _pywrapcp.RoutingDimension_SetSpanCostCoefficientForAllVehicles(self, coefficient)

    def SetSlackCostCoefficientForVehicle(self, coefficient, vehicle):
        """
        Sets a cost proportional to the dimension total slack on a given vehicle,
        or on all vehicles at once. "coefficient" must be nonnegative.
        This is handy to model costs only proportional to idle time when the
        dimension represents time.
        The cost for a vehicle is
          slack_cost = coefficient *
                (dimension end value - dimension start value - total_transit).
        """
        return _pywrapcp.RoutingDimension_SetSlackCostCoefficientForVehicle(self, coefficient, vehicle)

    def SetSlackCostCoefficientForAllVehicles(self, coefficient):
        return _pywrapcp.RoutingDimension_SetSlackCostCoefficientForAllVehicles(self, coefficient)

    def SetGlobalSpanCostCoefficient(self, coefficient):
        """
        Sets a cost proportional to the *global* dimension span, that is the
        difference between the largest value of route end cumul variables and
        the smallest value of route start cumul variables.
        In other words:
        global_span_cost =
          coefficient * (Max(dimension end value) - Min(dimension start value)).
        """
        return _pywrapcp.RoutingDimension_SetGlobalSpanCostCoefficient(self, coefficient)

    def SetCumulVarSoftUpperBound(self, index, upper_bound, coefficient):
        """
        Sets a soft upper bound to the cumul variable of a given variable index.
        If the value of the cumul variable is greater than the bound, a cost
        proportional to the difference between this value and the bound is added
        to the cost function of the model:
          cumulVar <= upper_bound -> cost = 0
           cumulVar > upper_bound -> cost = coefficient * (cumulVar - upper_bound)
        This is also handy to model tardiness costs when the dimension represents
        time.
        """
        return _pywrapcp.RoutingDimension_SetCumulVarSoftUpperBound(self, index, upper_bound, coefficient)

    def HasCumulVarSoftUpperBound(self, index):
        """
        Returns true if a soft upper bound has been set for a given variable
        index.
        """
        return _pywrapcp.RoutingDimension_HasCumulVarSoftUpperBound(self, index)

    def GetCumulVarSoftUpperBound(self, index):
        """
        Returns the soft upper bound of a cumul variable for a given variable
        index. The "hard" upper bound of the variable is returned if no soft upper
        bound has been set.
        """
        return _pywrapcp.RoutingDimension_GetCumulVarSoftUpperBound(self, index)

    def GetCumulVarSoftUpperBoundCoefficient(self, index):
        """
        Returns the cost coefficient of the soft upper bound of a cumul variable
        for a given variable index. If no soft upper bound has been set, 0 is
        returned.
        """
        return _pywrapcp.RoutingDimension_GetCumulVarSoftUpperBoundCoefficient(self, index)

    def SetCumulVarSoftLowerBound(self, index, lower_bound, coefficient):
        """
        Sets a soft lower bound to the cumul variable of a given variable index.
        If the value of the cumul variable is less than the bound, a cost
        proportional to the difference between this value and the bound is added
        to the cost function of the model:
          cumulVar > lower_bound -> cost = 0
          cumulVar <= lower_bound -> cost = coefficient * (lower_bound -
                      cumulVar).
        This is also handy to model earliness costs when the dimension represents
        time.
        """
        return _pywrapcp.RoutingDimension_SetCumulVarSoftLowerBound(self, index, lower_bound, coefficient)

    def HasCumulVarSoftLowerBound(self, index):
        """
        Returns true if a soft lower bound has been set for a given variable
        index.
        """
        return _pywrapcp.RoutingDimension_HasCumulVarSoftLowerBound(self, index)

    def GetCumulVarSoftLowerBound(self, index):
        """
        Returns the soft lower bound of a cumul variable for a given variable
        index. The "hard" lower bound of the variable is returned if no soft lower
        bound has been set.
        """
        return _pywrapcp.RoutingDimension_GetCumulVarSoftLowerBound(self, index)

    def GetCumulVarSoftLowerBoundCoefficient(self, index):
        """
        Returns the cost coefficient of the soft lower bound of a cumul variable
        for a given variable index. If no soft lower bound has been set, 0 is
        returned.
        """
        return _pywrapcp.RoutingDimension_GetCumulVarSoftLowerBoundCoefficient(self, index)

    def SetBreakIntervalsOfVehicle(self, breaks, vehicle, node_visit_transits):
        """
        Sets the breaks for a given vehicle. Breaks are represented by
        IntervalVars. They may interrupt transits between nodes and increase
        the value of corresponding slack variables.
        A break may take place before the start of a vehicle, after the end of
        a vehicle, or during a travel i -> j.

        In that case, the interval [break.Start(), break.End()) must be a subset
        of [CumulVar(i) + pre_travel(i, j), CumulVar(j) - post_travel(i, j)). In
        other words, a break may not overlap any node n's visit, given by
        [CumulVar(n) - post_travel(_, n), CumulVar(n) + pre_travel(n, _)).
        This formula considers post_travel(_, start) and pre_travel(end, _) to be
        0; pre_travel will never be called on any (_, start) and post_travel will
        never we called on any (end, _). If pre_travel_evaluator or
        post_travel_evaluator is -1, it will be taken as a function that always
        returns 0.
        Deprecated, sets pre_travel(i, j) = node_visit_transit[i].
        """
        return _pywrapcp.RoutingDimension_SetBreakIntervalsOfVehicle(self, breaks, vehicle, node_visit_transits)

    def SetBreakDistanceDurationOfVehicle(self, distance, duration, vehicle):
        """
        With breaks supposed to be consecutive, this forces the distance between
        breaks of size at least minimum_break_duration to be at most distance.
        This supposes that the time until route start and after route end are
        infinite breaks.
        """
        return _pywrapcp.RoutingDimension_SetBreakDistanceDurationOfVehicle(self, distance, duration, vehicle)

    def InitializeBreaks(self):
        """
        Sets up vehicle_break_intervals_, vehicle_break_distance_duration_,
        pre_travel_evaluators and post_travel_evaluators.
        """
        return _pywrapcp.RoutingDimension_InitializeBreaks(self)

    def HasBreakConstraints(self):
        """ Returns true if any break interval or break distance was defined."""
        return _pywrapcp.RoutingDimension_HasBreakConstraints(self)

    def GetPreTravelEvaluatorOfVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetPreTravelEvaluatorOfVehicle(self, vehicle)

    def GetPostTravelEvaluatorOfVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetPostTravelEvaluatorOfVehicle(self, vehicle)

    def base_dimension(self):
        """ Returns the parent in the dependency tree if any or nullptr otherwise."""
        return _pywrapcp.RoutingDimension_base_dimension(self)

    def ShortestTransitionSlack(self, node):
        """
        It makes sense to use the function only for self-dependent dimension.
        For such dimensions the value of the slack of a node determines the
        transition cost of the next transit. Provided that
          1. cumul[node] is fixed,
          2. next[node] and next[next[node]] (if exists) are fixed,
        the value of slack[node] for which cumul[next[node]] + transit[next[node]]
        is minimized can be found in O(1) using this function.
        """
        return _pywrapcp.RoutingDimension_ShortestTransitionSlack(self, node)

    def name(self):
        """ Returns the name of the dimension."""
        return _pywrapcp.RoutingDimension_name(self)

    def SetPickupToDeliveryLimitFunctionForPair(self, limit_function, pair_index):
        return _pywrapcp.RoutingDimension_SetPickupToDeliveryLimitFunctionForPair(self, limit_function, pair_index)

    def HasPickupToDeliveryLimits(self):
        return _pywrapcp.RoutingDimension_HasPickupToDeliveryLimits(self)

    def AddNodePrecedence(self, first_node, second_node, offset):
        return _pywrapcp.RoutingDimension_AddNodePrecedence(self, first_node, second_node, offset)

    def GetSpanUpperBoundForVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetSpanUpperBoundForVehicle(self, vehicle)

    def GetSpanCostCoefficientForVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetSpanCostCoefficientForVehicle(self, vehicle)

    def GetSlackCostCoefficientForVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetSlackCostCoefficientForVehicle(self, vehicle)

    def global_span_cost_coefficient(self):
        return _pywrapcp.RoutingDimension_global_span_cost_coefficient(self)

    def GetGlobalOptimizerOffset(self):
        return _pywrapcp.RoutingDimension_GetGlobalOptimizerOffset(self)

    def GetLocalOptimizerOffsetForVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetLocalOptimizerOffsetForVehicle(self, vehicle)

    def SetSoftSpanUpperBoundForVehicle(self, bound_cost, vehicle):
        """
        If the span of vehicle on this dimension is larger than bound,
        the cost will be increased by cost * (span - bound).
        """
        return _pywrapcp.RoutingDimension_SetSoftSpanUpperBoundForVehicle(self, bound_cost, vehicle)

    def HasSoftSpanUpperBounds(self):
        return _pywrapcp.RoutingDimension_HasSoftSpanUpperBounds(self)

    def GetSoftSpanUpperBoundForVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetSoftSpanUpperBoundForVehicle(self, vehicle)

    def SetQuadraticCostSoftSpanUpperBoundForVehicle(self, bound_cost, vehicle):
        """
        If the span of vehicle on this dimension is larger than bound,
        the cost will be increased by cost * (span - bound)^2.
        """
        return _pywrapcp.RoutingDimension_SetQuadraticCostSoftSpanUpperBoundForVehicle(self, bound_cost, vehicle)

    def HasQuadraticCostSoftSpanUpperBounds(self):
        return _pywrapcp.RoutingDimension_HasQuadraticCostSoftSpanUpperBounds(self)

    def GetQuadraticCostSoftSpanUpperBoundForVehicle(self, vehicle):
        return _pywrapcp.RoutingDimension_GetQuadraticCostSoftSpanUpperBoundForVehicle(self, vehicle)