from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdPlatformInfo(BaseSchema):
    """
    This object contains python version and implementation details.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'name': {'type': 'string', 'description': "Name of the platform as returned by 'sys.platform'."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, name=None, update_ids_from_dap=False, **kwargs):
        """
        :param string name: Name of the platform as returned by 'sys.platform'.
        """
        self.name = name
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        name = self.name
        dct = {}
        if name is not None:
            dct['name'] = name
        dct.update(self.kwargs)
        return dct