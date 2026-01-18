from __future__ import annotations
import logging # isort:skip
from functools import wraps
from .bases import Property
class validate:
    """ Control validation of bokeh properties

    This can be used as a context manager, or as a normal callable

    Args:
        value (bool) : Whether validation should occur or not

    Example:
        .. code-block:: python

            with validate(False):  # do no validate while within this block
                pass

            validate(False)  # don't validate ever

    See Also:
        :func:`~bokeh.core.property.bases.validation_on`: check the state of validation

        :func:`~bokeh.core.properties.without_property_validation`: function decorator

    """

    def __init__(self, value) -> None:
        self.old = Property._should_validate
        Property._should_validate = value

    def __enter__(self):
        pass

    def __exit__(self, typ, value, traceback):
        Property._should_validate = self.old