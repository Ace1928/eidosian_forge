from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdProcessInfo(BaseSchema):
    """
    This object contains python process details.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'pid': {'type': 'integer', 'description': 'Process ID for the current process.'}, 'ppid': {'type': 'integer', 'description': 'Parent Process ID for the current process.'}, 'executable': {'type': 'string', 'description': "Path to the executable as returned by 'sys.executable'."}, 'bitness': {'type': 'integer', 'description': 'Integer value indicating the bitness of the current process.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, pid=None, ppid=None, executable=None, bitness=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer pid: Process ID for the current process.
        :param integer ppid: Parent Process ID for the current process.
        :param string executable: Path to the executable as returned by 'sys.executable'.
        :param integer bitness: Integer value indicating the bitness of the current process.
        """
        self.pid = pid
        self.ppid = ppid
        self.executable = executable
        self.bitness = bitness
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        pid = self.pid
        ppid = self.ppid
        executable = self.executable
        bitness = self.bitness
        dct = {}
        if pid is not None:
            dct['pid'] = pid
        if ppid is not None:
            dct['ppid'] = ppid
        if executable is not None:
            dct['executable'] = executable
        if bitness is not None:
            dct['bitness'] = bitness
        dct.update(self.kwargs)
        return dct