from __future__ import annotations
from typing_extensions import TypedDict
class BatchListParams(TypedDict, total=False):
    after: str
    'A cursor for use in pagination.\n\n    `after` is an object ID that defines your place in the list. For instance, if\n    you make a list request and receive 100 objects, ending with obj_foo, your\n    subsequent call can include after=obj_foo in order to fetch the next page of the\n    list.\n    '
    limit: int
    'A limit on the number of objects to be returned.\n\n    Limit can range between 1 and 100, and the default is 20.\n    '