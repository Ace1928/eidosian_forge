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
def __dp(self):
    """
        Get the I{default} port if defined in the I{options}.

        @return: A L{MethodSelector} for the I{default} port.
        @rtype: L{MethodSelector}.

        """
    dp = self.__client.options.port
    if dp is not None:
        return self.__find(dp)