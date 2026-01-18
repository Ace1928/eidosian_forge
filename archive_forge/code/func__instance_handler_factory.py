import datetime
from functools import partial
import logging
def _instance_handler_factory(handler):
    """ Get the instance factory of an Instance or TraitInstance
    """
    from traits.api import BaseInstance, DefaultValue, TraitInstance
    if isinstance(handler, TraitInstance):
        return handler.aClass
    elif isinstance(handler, BaseInstance):
        if handler.default_value_type == DefaultValue.callable_and_args:
            default_value_getter, args, kwargs = handler.default_value
            return lambda: default_value_getter(*args, **kwargs)
        else:
            return handler.default_value
    else:
        msg = 'handler should be TraitInstance or BaseInstance, but got {}'
        raise ValueError(msg.format(repr(handler)))