import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class ControllerManager(EventDispatcher):
    """High level interface for managing game Controllers.

    This class provides a convenient way to handle the
    connection and disconnection of devices. A list of all
    connected Controllers can be queried at any time with the
    `get_controllers` method. For hot-plugging, events are
    dispatched for `on_connect` and `on_disconnect`.
    To use the ControllerManager, first make an instance::

        controller_man = pyglet.input.ControllerManager()

    At the start of your game, query for any Controllers
    that are already connected::

        controllers = controller_man.get_controllers()

    To handle Controllers that are connected or disconnected
    after the start of your game, register handlers for the
    appropriate events::

        @controller_man.event
        def on_connect(controller):
            # code to handle newly connected
            # (or re-connected) controllers
            controller.open()
            print("Connect:", controller)

        @controller_man.event
        def on_disconnect(controller):
            # code to handle disconnected Controller
            print("Disconnect:", controller)

    .. versionadded:: 1.2
    """

    def get_controllers(self):
        """Get a list of all connected Controllers

        :rtype: list of :py:class:`Controller`
        """
        raise NotImplementedError

    def on_connect(self, controller):
        """A Controller has been connected. If this is
        a previously dissconnected Controller that is
        being re-connected, the same Controller instance
        will be returned.

        :Parameters:
            `controller` : :py:class:`Controller`
                An un-opened Controller instance.

        :event:
        """

    def on_disconnect(self, controller):
        """A Controller has been disconnected.

        :Parameters:
            `controller` : :py:class:`Controller`
                An un-opened Controller instance.

        :event:
        """