from abc import ABC
import collections
import enum
import functools
import logging
def _call_logging_exceptions(behavior, message, *args, **kwargs):
    try:
        return _EasyOutcome(Outcome.Kind.RETURNED, behavior(*args, **kwargs), None)
    except Exception as e:
        _LOGGER.exception(message)
        return _EasyOutcome(Outcome.Kind.RAISED, None, e)