import suds
from suds import *
import suds.bindings.binding
from suds.builder import Builder
import suds.cache
import suds.metrics as metrics
from suds.options import Options
from suds.plugin import PluginContainer
from suds.properties import Unskin
from suds.reader import DefinitionsReader
from suds.resolver import PathResolver
from suds.sax.document import Document
import suds.sax.parser
from suds.servicedefinition import ServiceDefinition
import suds.transport
import suds.transport.https
from suds.umx.basic import Basic as UmxBasic
from suds.wsdl import Definitions
from . import sudsobject
from http.cookiejar import CookieJar
from copy import deepcopy
import http.client
from logging import getLogger
class ServiceSelector:
    """
    The B{service} selector is used to select a web service.

    Most WSDLs only define a single service in which case access by subscript
    is passed through to a L{PortSelector}. This is also the behavior when a
    I{default} service has been specified. In cases where multiple services
    have been defined and no default has been specified, the service is found
    by name (or index) and a L{PortSelector} for the service is returned. In
    all cases, attribute access is forwarded to the L{PortSelector} for either
    the I{first} service or the I{default} service (when specified).

    @ivar __client: A suds client.
    @type __client: L{Client}
    @ivar __services: A list of I{WSDL} services.
    @type __services: list

    """

    def __init__(self, client, services):
        """
        @param client: A suds client.
        @type client: L{Client}
        @param services: A list of I{WSDL} services.
        @type services: list

        """
        self.__client = client
        self.__services = services

    def __getattr__(self, name):
        """
        Attribute access is forwarded to the L{PortSelector}.

        Uses the I{default} service if specified or the I{first} service
        otherwise.

        @param name: Method name.
        @type name: str
        @return: A L{PortSelector}.
        @rtype: L{PortSelector}.

        """
        default = self.__ds()
        if default is None:
            port = self.__find(0)
        else:
            port = default
        return getattr(port, name)

    def __getitem__(self, name):
        """
        Provides I{service} selection by name (string) or index (integer).

        In cases where only a single service is defined or a I{default} has
        been specified, the request is forwarded to the L{PortSelector}.

        @param name: The name (or index) of a service.
        @type name: int|str
        @return: A L{PortSelector} for the specified service.
        @rtype: L{PortSelector}.

        """
        if len(self.__services) == 1:
            port = self.__find(0)
            return port[name]
        default = self.__ds()
        if default is not None:
            port = default
            return port[name]
        return self.__find(name)

    def __find(self, name):
        """
        Find a I{service} by name (string) or index (integer).

        @param name: The name (or index) of a service.
        @type name: int|str
        @return: A L{PortSelector} for the found service.
        @rtype: L{PortSelector}.

        """
        service = None
        if not self.__services:
            raise Exception('No services defined')
        if isinstance(name, int):
            try:
                service = self.__services[name]
                name = service.name
            except IndexError:
                raise ServiceNotFound('at [%d]' % (name,))
        else:
            for s in self.__services:
                if name == s.name:
                    service = s
                    break
        if service is None:
            raise ServiceNotFound(name)
        return PortSelector(self.__client, service.ports, name)

    def __ds(self):
        """
        Get the I{default} service if defined in the I{options}.

        @return: A L{PortSelector} for the I{default} service.
        @rtype: L{PortSelector}.

        """
        ds = self.__client.options.service
        if ds is not None:
            return self.__find(ds)