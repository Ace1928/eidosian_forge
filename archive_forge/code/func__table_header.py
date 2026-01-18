import json
import logging
import re
import uuid
import warnings
from base64 import b64encode
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import UNPKG_DT_BUNDLE_CSS, UNPKG_DT_BUNDLE_URL
from .version import __version__ as itables_version
from IPython.display import HTML, display
import itables.options as opt
from .datatables_format import datatables_rows
from .downsample import downsample
from .utils import read_package_file
def _table_header(df, table_id, show_index, classes, style, tags, footer, column_filters):
    """This function returns the HTML table header. Rows are not included."""
    pattern = re.compile('.*<thead>(.*)</thead>', flags=re.MULTILINE | re.DOTALL)
    try:
        html_header = df.head(0).to_html(escape=False)
    except AttributeError:
        html_header = pd.DataFrame(data=[], columns=df.columns).to_html()
    match = pattern.match(html_header)
    thead = match.groups()[0]
    if not show_index and len(df.columns):
        thead = thead.replace('<th></th>', '', 1)
    loading = '<td>Loading... (need <a href=https://mwouts.github.io/itables/troubleshooting.html>help</a>?)</td>'
    tbody = '<tr>{}</tr>'.format(loading)
    if style:
        style = 'style="{}"'.format(style)
    else:
        style = ''
    header = '<thead>{}</thead>'.format(_flat_header(df, show_index) if column_filters == 'header' else thead)
    if column_filters == 'footer':
        footer = '<tfoot>{}</tfoot>'.format(_flat_header(df, show_index))
    elif footer:
        footer = '<tfoot>{}</tfoot>'.format(_tfoot_from_thead(thead))
    else:
        footer = ''
    return '<table id="{table_id}" class="{classes}" data-quarto-disable-processing="true" {style}>\n{tags}{header}<tbody>{tbody}</tbody>\n{footer}\n</table>'.format(table_id=table_id, classes=classes, style=style, tags=tags, header=header, tbody=tbody, footer=footer)