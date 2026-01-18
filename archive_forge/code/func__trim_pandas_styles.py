from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
def _trim_pandas_styles(styles: list[M]) -> list[M]:
    """Filter out empty styles.

    Every cell will have a class, but the list of props
    may just be [['', '']].

    Parameters
    ----------
    styles : list
        pandas.Styler translated styles.

    """
    return [x for x in styles if any((any(y) for y in x['props']))]