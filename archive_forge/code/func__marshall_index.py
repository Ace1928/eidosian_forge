from __future__ import annotations
from typing import TYPE_CHECKING, Any
from streamlit import type_util
from streamlit.elements.lib import pandas_styler_utils
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
def _marshall_index(proto: ArrowTableProto, index: Index) -> None:
    """Marshall pandas.DataFrame index into an ArrowTable proto.

    Parameters
    ----------
    proto : proto.ArrowTable
        Output. The protobuf for a Streamlit ArrowTable proto.

    index : pd.Index
        Index to use for resulting frame.
        Will default to RangeIndex (0, 1, 2, ..., n) if no index is provided.

    """
    import pandas as pd
    index = map(type_util.maybe_tuple_to_list, index.values)
    index_df = pd.DataFrame(index)
    proto.index = type_util.data_frame_to_bytes(index_df)