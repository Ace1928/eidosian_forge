from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class DataBreakpoint(BaseSchema):
    """
    Properties of a data breakpoint passed to the setDataBreakpoints request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'dataId': {'type': 'string', 'description': 'An id representing the data. This id is returned from the dataBreakpointInfo request.'}, 'accessType': {'description': 'The access type of the data.', 'type': 'DataBreakpointAccessType'}, 'condition': {'type': 'string', 'description': 'An optional expression for conditional breakpoints.'}, 'hitCondition': {'type': 'string', 'description': 'An optional expression that controls how many hits of the breakpoint are ignored.\nThe backend is expected to interpret the expression as needed.'}}
    __refs__ = set(['accessType'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, dataId, accessType=None, condition=None, hitCondition=None, update_ids_from_dap=False, **kwargs):
        """
        :param string dataId: An id representing the data. This id is returned from the dataBreakpointInfo request.
        :param DataBreakpointAccessType accessType: The access type of the data.
        :param string condition: An optional expression for conditional breakpoints.
        :param string hitCondition: An optional expression that controls how many hits of the breakpoint are ignored.
        The backend is expected to interpret the expression as needed.
        """
        self.dataId = dataId
        if accessType is not None:
            assert accessType in DataBreakpointAccessType.VALID_VALUES
        self.accessType = accessType
        self.condition = condition
        self.hitCondition = hitCondition
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        dataId = self.dataId
        accessType = self.accessType
        condition = self.condition
        hitCondition = self.hitCondition
        dct = {'dataId': dataId}
        if accessType is not None:
            dct['accessType'] = accessType
        if condition is not None:
            dct['condition'] = condition
        if hitCondition is not None:
            dct['hitCondition'] = hitCondition
        dct.update(self.kwargs)
        return dct