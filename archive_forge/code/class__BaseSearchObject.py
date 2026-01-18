from typing import Tuple, Union
from Bio.SearchIO._utils import getattr_str
class _BaseSearchObject:
    """Abstract class for SearchIO objects."""
    _NON_STICKY_ATTRS: Union[Tuple, Tuple[str]] = ()

    def _transfer_attrs(self, obj):
        """Transfer instance attributes to the given object (PRIVATE).

        This method is used to transfer attributes set externally (for example
        using ``setattr``) to a new object created from this one (for example
        from slicing).

        The reason this method is necessary is because different parsers will
        set different attributes for each QueryResult, Hit, HSP, or HSPFragment
        objects, depending on the attributes they found in the search output
        file. Ideally, we want these attributes to 'stick' with any new instance
        object created from the original one.

        """
        for attr in self.__dict__:
            if attr not in self._NON_STICKY_ATTRS:
                setattr(obj, attr, self.__dict__[attr])