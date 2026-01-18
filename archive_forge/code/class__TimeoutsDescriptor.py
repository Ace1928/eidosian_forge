from typing import TYPE_CHECKING
class _TimeoutsDescriptor:
    """Get or set the value of the attributes listed below.

    _implicit_wait _page_load _script

    This does not set the value on the remote end.
    """

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls) -> float:
        return getattr(obj, self.name) / 1000

    def __set__(self, obj, value) -> None:
        converted_value = getattr(obj, '_convert')(value)
        setattr(obj, self.name, converted_value)