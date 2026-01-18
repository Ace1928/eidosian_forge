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
def __find(self, name):
    """
        Find a I{port} by name (string) or index (integer).

        @param name: The name (or index) of a port.
        @type name: int|str
        @return: A L{MethodSelector} for the found port.
        @rtype: L{MethodSelector}.

        """
    port = None
    if not self.__ports:
        raise Exception('No ports defined: %s' % (self.__qn,))
    if isinstance(name, int):
        qn = '%s[%d]' % (self.__qn, name)
        try:
            port = self.__ports[name]
        except IndexError:
            raise PortNotFound(qn)
    else:
        qn = '.'.join((self.__qn, name))
        for p in self.__ports:
            if name == p.name:
                port = p
                break
    if port is None:
        raise PortNotFound(qn)
    qn = '.'.join((self.__qn, port.name))
    return MethodSelector(self.__client, port.methods, qn)