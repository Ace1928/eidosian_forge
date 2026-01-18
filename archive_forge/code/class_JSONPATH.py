from .array import ARRAY
from .array import array as _pg_array
from .operators import ASTEXT
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import DELETE_PATH
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from .operators import JSONPATH_ASTEXT
from .operators import PATH_EXISTS
from .operators import PATH_MATCH
from ... import types as sqltypes
from ...sql import cast
class JSONPATH(JSONPathType):
    """JSON Path Type.

    This is usually required to cast literal values to json path when using
    json search like function, such as ``jsonb_path_query_array`` or
    ``jsonb_path_exists``::

        stmt = sa.select(
            sa.func.jsonb_path_query_array(
                table.c.jsonb_col, cast("$.address.id", JSONPATH)
            )
        )

    """
    __visit_name__ = 'JSONPATH'