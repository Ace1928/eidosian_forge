from __future__ import annotations
import csv
import datetime
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any
import filelock
import huggingface_hub
from gradio_client import utils as client_utils
from gradio_client.documentation import document
import gradio as gr
from gradio import utils
@document()
class SimpleCSVLogger(FlaggingCallback):
    """
    A simplified implementation of the FlaggingCallback abstract class
    provided for illustrative purposes.  Each flagged sample (both the input and output data)
    is logged to a CSV file on the machine running the gradio app.
    Example:
        import gradio as gr
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            flagging_callback=SimpleCSVLogger())
    """

    def __init__(self):
        pass

    def setup(self, components: list[Component], flagging_dir: str | Path):
        self.components = components
        self.flagging_dir = flagging_dir
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(self, flag_data: list[Any], flag_option: str='', username: str | None=None) -> int:
        flagging_dir = self.flagging_dir
        log_filepath = Path(flagging_dir) / 'log.csv'
        csv_data = []
        for component, sample in zip(self.components, flag_data):
            save_dir = Path(flagging_dir) / client_utils.strip_invalid_filename_characters(component.label or '')
            save_dir.mkdir(exist_ok=True)
            csv_data.append(component.flag(sample, save_dir))
        with open(log_filepath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(utils.sanitize_list_for_csv(csv_data))
        with open(log_filepath) as csvfile:
            line_count = len(list(csv.reader(csvfile))) - 1
        return line_count