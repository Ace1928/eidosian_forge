from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
def _use_display_values(df: DataFrame, styles: Mapping[str, Any]) -> DataFrame:
    """Create a new pandas.DataFrame where display values are used instead of original ones.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with original values.

    styles : dict
        pandas.Styler translated styles.

    """
    import re
    new_df = df.astype(str)
    cell_selector_regex = re.compile('row(\\d+)_col(\\d+)')
    if 'body' in styles:
        rows = styles['body']
        for row in rows:
            for cell in row:
                if 'id' in cell:
                    if (match := cell_selector_regex.match(cell['id'])):
                        r, c = map(int, match.groups())
                        new_df.iat[r, c] = str(cell['display_value'])
    return new_df