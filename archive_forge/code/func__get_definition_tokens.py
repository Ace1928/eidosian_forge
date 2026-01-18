from importlib import import_module
import inspect
import io
import token
import tokenize
import traceback
from sphinx.ext.autodoc import ClassLevelDocumenter
from sphinx.util import logging
from traits.has_traits import MetaHasTraits
from traits.trait_type import TraitType
from traits.traits import generic_trait
def _get_definition_tokens(tokens):
    """ Given the tokens, extracts the definition tokens.

    Parameters
    ----------
    tokens : iterator
        An iterator producing tokens.

    Returns
    -------
    A list of tokens for the definition.
    """
    definition_tokens = []
    first_line = None
    for type, name, start, stop, line_text in tokens:
        if first_line is None:
            first_line = start[0]
        if type == token.NEWLINE:
            break
        item = (type, name, (start[0] - first_line + 1, start[1]), (stop[0] - first_line + 1, stop[1]), line_text)
        definition_tokens.append(item)
    return definition_tokens