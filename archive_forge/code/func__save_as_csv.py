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
@staticmethod
def _save_as_csv(data_file: Path, headers: list[str], row: list[Any]) -> int:
    """Save data as CSV and return the sample name (row number)."""
    is_new = not data_file.exists()
    with data_file.open('a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(utils.sanitize_list_for_csv(headers))
        writer.writerow(utils.sanitize_list_for_csv(row))
    with data_file.open(encoding='utf-8') as csvfile:
        return sum((1 for _ in csv.reader(csvfile))) - 1