import os
import urllib
from typing import AnyStr
from twisted.internet import protocol
from twisted.logger import Logger
from twisted.python import filepath
from twisted.spread import pb
from twisted.web import http, resource, server, static
class FilteredScript(CGIScript):
    """
    I am a special version of a CGI script, that uses a specific executable.

    This is useful for interfacing with other scripting languages that adhere
    to the CGI standard. My C{filter} attribute specifies what executable to
    run, and my C{filename} init parameter describes which script to pass to
    the first argument of that script.

    To customize me for a particular location of a CGI interpreter, override
    C{filter}.

    @type filter: L{str}
    @ivar filter: The absolute path to the executable.
    """
    filter = '/usr/bin/cat'

    def runProcess(self, env, request, qargs=[]):
        """
        Run a script through the C{filter} executable.

        @type env: A L{dict} of L{str}, or L{None}
        @param env: The environment variables to pass to the process that will
            get spawned. See
            L{twisted.internet.interfaces.IReactorProcess.spawnProcess}
            for more information about environments and process creation.

        @type request: L{twisted.web.http.Request}
        @param request: An HTTP request.

        @type qargs: A L{list} of L{str}
        @param qargs: The command line arguments to pass to the process that
            will get spawned.
        """
        p = CGIProcessProtocol(request)
        self._reactor.spawnProcess(p, self.filter, [self.filter, self.filename] + qargs, env, os.path.dirname(self.filename))