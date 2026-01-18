from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
def _marshall_styles(proto: ArrowProto, styler: Styler, styles: Mapping[str, Any]) -> None:
    """Marshall pandas.Styler styles into an Arrow proto.

    Parameters
    ----------
    proto : proto.Arrow
        Output. The protobuf for Streamlit Arrow proto.

    styler : pandas.Styler
        Helps style a DataFrame or Series according to the data with HTML and CSS.

    styles : dict
        pandas.Styler translated styles.

    """
    css_rules = []
    if 'table_styles' in styles:
        table_styles = styles['table_styles']
        table_styles = _trim_pandas_styles(table_styles)
        for style in table_styles:
            rule = _pandas_style_to_css('table_styles', style, styler.uuid, separator=' ')
            css_rules.append(rule)
    if 'cellstyle' in styles:
        cellstyle = styles['cellstyle']
        cellstyle = _trim_pandas_styles(cellstyle)
        for style in cellstyle:
            rule = _pandas_style_to_css('cell_style', style, styler.uuid)
            css_rules.append(rule)
    if len(css_rules) > 0:
        proto.styler.styles = '\n'.join(css_rules)