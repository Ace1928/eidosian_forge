from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
class FactoryNotFoundError(AttributeError):
    """
    ``FactoryNotFound`` exception class.

    Raise when no matching factory could be found.
    """
    pass