from __future__ import annotations
import json
import re
from traitlets.log import get_logger
from nbformat import v3, validator
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode
from .nbbase import nbformat, nbformat_minor
def _warn_if_invalid(nb, version):
    """Log validation errors, if there are any."""
    from nbformat import ValidationError, validate
    try:
        validate(nb, version=version)
    except ValidationError as e:
        get_logger().error('Notebook JSON is not valid v%i: %s', version, e)