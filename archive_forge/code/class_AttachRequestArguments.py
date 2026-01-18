from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class AttachRequestArguments(BaseSchema):
    """
    Arguments for 'attach' request. Additional attributes are implementation specific.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'__restart': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': "Optional data from the previous, restarted session.\nThe data is sent as the 'restart' attribute of the 'terminated' event.\nThe client should leave the data intact."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, __restart=None, update_ids_from_dap=False, **kwargs):
        """
        :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] __restart: Optional data from the previous, restarted session.
        The data is sent as the 'restart' attribute of the 'terminated' event.
        The client should leave the data intact.
        """
        self.__restart = __restart
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        __restart = self.__restart
        dct = {}
        if __restart is not None:
            dct['__restart'] = __restart
        dct.update(self.kwargs)
        return dct