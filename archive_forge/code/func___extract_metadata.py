from __future__ import annotations
import warnings
from typing import (
import numpy as np
import semantic_version
from gradio_client.documentation import document
from gradio.components import Component
from gradio.data_classes import GradioModel
from gradio.events import Events
@staticmethod
def __extract_metadata(df: Styler) -> dict[str, list[list]]:
    metadata = {'display_value': [], 'styling': []}
    style_data = df._compute()._translate(None, None)
    cell_styles = style_data.get('cellstyle', [])
    for i in range(len(style_data['body'])):
        metadata['display_value'].append([])
        metadata['styling'].append([])
        for j in range(len(style_data['body'][i])):
            cell_type = style_data['body'][i][j]['type']
            if cell_type != 'td':
                continue
            display_value = style_data['body'][i][j]['display_value']
            cell_id = style_data['body'][i][j]['id']
            styles_str = Dataframe.__get_cell_style(cell_id, cell_styles)
            metadata['display_value'][i].append(display_value)
            metadata['styling'][i].append(styles_str)
    return metadata