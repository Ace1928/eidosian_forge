import threading
from paramiko import util
from paramiko.common import (
class SubsystemHandler(threading.Thread):
    """
    Handler for a subsystem in server mode.  If you create a subclass of this
    class and pass it to `.Transport.set_subsystem_handler`, an object of this
    class will be created for each request for this subsystem.  Each new object
    will be executed within its own new thread by calling `start_subsystem`.
    When that method completes, the channel is closed.

    For example, if you made a subclass ``MP3Handler`` and registered it as the
    handler for subsystem ``"mp3"``, then whenever a client has successfully
    authenticated and requests subsystem ``"mp3"``, an object of class
    ``MP3Handler`` will be created, and `start_subsystem` will be called on
    it from a new thread.
    """

    def __init__(self, channel, name, server):
        """
        Create a new handler for a channel.  This is used by `.ServerInterface`
        to start up a new handler when a channel requests this subsystem.  You
        don't need to override this method, but if you do, be sure to pass the
        ``channel`` and ``name`` parameters through to the original
        ``__init__`` method here.

        :param .Channel channel: the channel associated with this
            subsystem request.
        :param str name: name of the requested subsystem.
        :param .ServerInterface server:
            the server object for the session that started this subsystem
        """
        threading.Thread.__init__(self, target=self._run)
        self.__channel = channel
        self.__transport = channel.get_transport()
        self.__name = name
        self.__server = server

    def get_server(self):
        """
        Return the `.ServerInterface` object associated with this channel and
        subsystem.
        """
        return self.__server

    def _run(self):
        try:
            self.__transport._log(DEBUG, 'Starting handler for subsystem {}'.format(self.__name))
            self.start_subsystem(self.__name, self.__transport, self.__channel)
        except Exception as e:
            self.__transport._log(ERROR, 'Exception in subsystem handler for "{}": {}'.format(self.__name, e))
            self.__transport._log(ERROR, util.tb_strings())
        try:
            self.finish_subsystem()
        except:
            pass

    def start_subsystem(self, name, transport, channel):
        """
        Process an ssh subsystem in server mode.  This method is called on a
        new object (and in a new thread) for each subsystem request.  It is
        assumed that all subsystem logic will take place here, and when the
        subsystem is finished, this method will return.  After this method
        returns, the channel is closed.

        The combination of ``transport`` and ``channel`` are unique; this
        handler corresponds to exactly one `.Channel` on one `.Transport`.

        .. note::
            It is the responsibility of this method to exit if the underlying
            `.Transport` is closed.  This can be done by checking
            `.Transport.is_active` or noticing an EOF on the `.Channel`.  If
            this method loops forever without checking for this case, your
            Python interpreter may refuse to exit because this thread will
            still be running.

        :param str name: name of the requested subsystem.
        :param .Transport transport: the server-mode `.Transport`.
        :param .Channel channel: the channel associated with this subsystem
            request.
        """
        pass

    def finish_subsystem(self):
        """
        Perform any cleanup at the end of a subsystem.  The default
        implementation just closes the channel.

        .. versionadded:: 1.1
        """
        self.__channel.close()