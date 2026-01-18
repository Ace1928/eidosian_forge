import builtins
import types
import sys
from inspect import getmembers
from webob.exc import HTTPFound
from .util import iscontroller, _cfg
class PecanHook(object):
    """
    A base class for Pecan hooks. Inherit from this class to create your
    own hooks. Set a priority on a hook by setting the ``priority``
    attribute for the hook, which defaults to 100.
    """
    priority = 100

    def on_route(self, state):
        """
        Override this method to create a hook that gets called upon
        the start of routing.

        :param state: The Pecan ``state`` object for the current request.
        """
        return

    def before(self, state):
        """
        Override this method to create a hook that gets called after
        routing, but before the request gets passed to your controller.

        :param state: The Pecan ``state`` object for the current request.
        """
        return

    def after(self, state):
        """
        Override this method to create a hook that gets called after
        the request has been handled by the controller.

        :param state: The Pecan ``state`` object for the current request.
        """
        return

    def on_error(self, state, e):
        """
        Override this method to create a hook that gets called upon
        an exception being raised in your controller.

        :param state: The Pecan ``state`` object for the current request.
        :param e: The ``Exception`` object that was raised.
        """
        return