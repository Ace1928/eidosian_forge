import logging
from pathlib import Path
from typing import Optional
import srsly
import typer
from wasabi import msg
from .. import util
from ..language import Language
from ..training.initialize import convert_vectors, init_nlp
from ._util import (
def _init_labels(nlp, output_path):
    for name, component in nlp.pipeline:
        if getattr(component, 'label_data', None) is not None:
            output_file = output_path / f'{name}.json'
            srsly.write_json(output_file, component.label_data)
            msg.good(f"Saving label data for component '{name}' to {output_file}")
        else:
            msg.info(f"No label data found for component '{name}'")