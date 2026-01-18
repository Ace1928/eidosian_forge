from typing import List, Optional, Union
def add_filter(self, flt: 'Filter') -> 'Query':
    """
        Add a numeric or geo filter to the query.
        **Currently only one of each filter is supported by the engine**

        - **flt**: A NumericFilter or GeoFilter object, used on a
        corresponding field
        """
    self._filters.append(flt)
    return self