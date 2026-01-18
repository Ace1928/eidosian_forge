from pyarrow.lib import Table
from pyarrow.compute import Expression, field
Filter rows of a table based on the provided expression.

    The result will be an output table with only the rows matching
    the provided expression.

    Parameters
    ----------
    table : Table or Dataset
        Table or Dataset that should be filtered.
    expression : Expression
        The expression on which rows should be filtered.

    Returns
    -------
    Table
    