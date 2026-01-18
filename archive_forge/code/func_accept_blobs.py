from __future__ import annotations
import dataclasses
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
import numpy as np
import PIL.Image
from gradio_client import file
from gradio_client.documentation import document
from typing_extensions import TypedDict
from gradio import image_utils, utils
from gradio.components.base import Component, server
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
@server
def accept_blobs(self, data: AcceptBlobs):
    """
        Accepts a dictionary of image blobs, where the keys are 'background', 'layers', and 'composite', and the values are binary file-like objects.
        """
    type = data.data['type']
    index = int(data.data['index']) if data.data['index'] and data.data['index'] != 'null' else None
    file = data.files[0][1]
    id = data.data['id']
    current = self.blob_storage.get(id, EditorDataBlobs(background=None, layers=[], composite=None))
    if type == 'layer' and index is not None:
        if index >= len(current.layers):
            current.layers.extend([None] * (index + 1 - len(current.layers)))
        current.layers[index] = file
    elif type == 'background':
        current.background = file
    elif type == 'composite':
        current.composite = file
    self.blob_storage[id] = current