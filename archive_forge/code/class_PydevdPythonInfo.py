from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdPythonInfo(BaseSchema):
    """
    This object contains python version and implementation details.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'version': {'type': 'string', 'description': 'Python version as a string in semver format: <major>.<minor>.<micro><releaselevel><serial>.'}, 'implementation': {'description': 'Python version as a string in this format <major>.<minor>.<micro><releaselevel><serial>.', 'type': 'PydevdPythonImplementationInfo'}}
    __refs__ = set(['implementation'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, version=None, implementation=None, update_ids_from_dap=False, **kwargs):
        """
        :param string version: Python version as a string in semver format: <major>.<minor>.<micro><releaselevel><serial>.
        :param PydevdPythonImplementationInfo implementation: Python version as a string in this format <major>.<minor>.<micro><releaselevel><serial>.
        """
        self.version = version
        if implementation is None:
            self.implementation = PydevdPythonImplementationInfo()
        else:
            self.implementation = PydevdPythonImplementationInfo(update_ids_from_dap=update_ids_from_dap, **implementation) if implementation.__class__ != PydevdPythonImplementationInfo else implementation
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        version = self.version
        implementation = self.implementation
        dct = {}
        if version is not None:
            dct['version'] = version
        if implementation is not None:
            dct['implementation'] = implementation.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct