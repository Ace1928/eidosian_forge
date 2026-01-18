from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class first_value(Reducer):
    """
    Selects the first value within the group according to sorting parameters
    """
    NAME = 'FIRST_VALUE'

    def __init__(self, field: str, *byfields: Union[Asc, Desc]) -> None:
        """
        Selects the first value of the given field within the group.

        ### Parameter

        - **field**: Source field used for the value
        - **byfields**: How to sort the results. This can be either the
            *class* of `aggregation.Asc` or `aggregation.Desc` in which
            case the field `field` is also used as the sort input.

            `byfields` can also be one or more *instances* of `Asc` or `Desc`
            indicating the sort order for these fields
        """
        fieldstrs = []
        if len(byfields) == 1 and isinstance(byfields[0], type) and issubclass(byfields[0], SortDirection):
            byfields = [byfields[0](field)]
        for f in byfields:
            fieldstrs += [f.field, f.DIRSTRING]
        args = [field]
        if fieldstrs:
            args += ['BY'] + fieldstrs
        super().__init__(*args)
        self._field = field