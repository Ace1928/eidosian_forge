from typing import TYPE_CHECKING
from typing import List
from typing import Optional
class _PageOrientationDescriptor:
    """PageOrientation descriptor which validates orientation of page."""
    ORIENTATION_VALUES = ['portrait', 'landscape']

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls) -> Optional[Orientation]:
        return obj._print_options.get(self.name, None)

    def __set__(self, obj, value) -> None:
        if value not in self.ORIENTATION_VALUES:
            raise ValueError(f'Orientation value must be one of {self.ORIENTATION_VALUES}')
        obj._print_options[self.name] = value