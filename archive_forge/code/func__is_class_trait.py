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
def _is_class_trait(name, cls):
    """ Check if the name is in the list of class defined traits of ``cls``.
    """
    return isinstance(cls, MetaHasTraits) and name in cls.__class_traits__ and (cls.__class_traits__[name] is not generic_trait)