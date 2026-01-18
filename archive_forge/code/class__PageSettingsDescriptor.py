from typing import TYPE_CHECKING
from typing import List
from typing import Optional
class _PageSettingsDescriptor:
    """Descriptor which validates `height` and 'width' of page."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls) -> Optional[float]:
        return obj._page.get(self.name, None)

    def __set__(self, obj, value) -> None:
        getattr(obj, '_validate_num_property')(self.name, value)
        obj._page[self.name] = value
        obj._print_options['page'] = obj._page