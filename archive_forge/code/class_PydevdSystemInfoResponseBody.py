from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdSystemInfoResponseBody(BaseSchema):
    """
    "body" of PydevdSystemInfoResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'python': {'description': 'Information about the python version running in the current process.', 'type': 'PydevdPythonInfo'}, 'platform': {'description': 'Information about the plarforn on which the current process is running.', 'type': 'PydevdPlatformInfo'}, 'process': {'description': 'Information about the current process.', 'type': 'PydevdProcessInfo'}, 'pydevd': {'description': 'Information about pydevd.', 'type': 'PydevdInfo'}}
    __refs__ = set(['python', 'platform', 'process', 'pydevd'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, python, platform, process, pydevd, update_ids_from_dap=False, **kwargs):
        """
        :param PydevdPythonInfo python: Information about the python version running in the current process.
        :param PydevdPlatformInfo platform: Information about the plarforn on which the current process is running.
        :param PydevdProcessInfo process: Information about the current process.
        :param PydevdInfo pydevd: Information about pydevd.
        """
        if python is None:
            self.python = PydevdPythonInfo()
        else:
            self.python = PydevdPythonInfo(update_ids_from_dap=update_ids_from_dap, **python) if python.__class__ != PydevdPythonInfo else python
        if platform is None:
            self.platform = PydevdPlatformInfo()
        else:
            self.platform = PydevdPlatformInfo(update_ids_from_dap=update_ids_from_dap, **platform) if platform.__class__ != PydevdPlatformInfo else platform
        if process is None:
            self.process = PydevdProcessInfo()
        else:
            self.process = PydevdProcessInfo(update_ids_from_dap=update_ids_from_dap, **process) if process.__class__ != PydevdProcessInfo else process
        if pydevd is None:
            self.pydevd = PydevdInfo()
        else:
            self.pydevd = PydevdInfo(update_ids_from_dap=update_ids_from_dap, **pydevd) if pydevd.__class__ != PydevdInfo else pydevd
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        python = self.python
        platform = self.platform
        process = self.process
        pydevd = self.pydevd
        dct = {'python': python.to_dict(update_ids_to_dap=update_ids_to_dap), 'platform': platform.to_dict(update_ids_to_dap=update_ids_to_dap), 'process': process.to_dict(update_ids_to_dap=update_ids_to_dap), 'pydevd': pydevd.to_dict(update_ids_to_dap=update_ids_to_dap)}
        dct.update(self.kwargs)
        return dct