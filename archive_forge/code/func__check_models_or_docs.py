from __future__ import annotations
import logging # isort:skip
from typing import (
from .. import __version__
from ..core.templates import (
from ..document.document import DEFAULT_TITLE, Document
from ..model import Model
from ..resources import Resources, ResourcesLike
from ..themes import Theme
from .bundle import Script, bundle_for_objs_and_resources
from .elements import html_page_for_render_items, script_for_render_items
from .util import (
from .wrappers import wrap_in_onload
def _check_models_or_docs(models: ModelLike | ModelLikeCollection) -> ModelLikeCollection:
    """

    """
    input_type_valid = False
    if isinstance(models, (Model, Document)):
        models = [models]
    if isinstance(models, Sequence) and all((isinstance(x, (Model, Document)) for x in models)):
        input_type_valid = True
    if isinstance(models, dict) and all((isinstance(x, str) for x in models.keys())) and all((isinstance(x, (Model, Document)) for x in models.values())):
        input_type_valid = True
    if not input_type_valid:
        raise ValueError('Input must be a Model, a Document, a Sequence of Models and Document, or a dictionary from string to Model and Document')
    return models