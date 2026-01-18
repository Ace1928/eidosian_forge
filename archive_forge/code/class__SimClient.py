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
class _SimClient(_SoapClient):
    """
    Loopback _SoapClient used for SOAP request/reply simulation.

    Used when a web service operation is invoked with injected SOAP request or
    reply data.

    """
    __injkey = '__inject'

    @classmethod
    def simulation(cls, kwargs):
        """Get whether injected data has been specified in I{kwargs}."""
        return _SimClient.__injkey in kwargs

    def invoke(self, args, kwargs):
        """
        Invoke a specified web service method.

        Uses an injected SOAP request/response instead of a regularly
        constructed/received one.

        Depending on how the ``nosend`` & ``retxml`` options are set, may do
        one of the following:
          * Return a constructed web service operation request without sending
            it to the web service.
          * Invoke the web service operation and return its SOAP reply XML.
          * Invoke the web service operation, process its results and return
            the Python object representing the returned value.

        @param args: Positional arguments for the method invoked.
        @type args: list|tuple
        @param kwargs: Keyword arguments for the method invoked.
        @type kwargs: dict
        @return: SOAP request, SOAP reply or a web service return value.
        @rtype: L{RequestContext}|I{builtin}|I{subclass of} L{Object}|I{bytes}|
            I{None}

        """
        simulation = kwargs.pop(self.__injkey)
        msg = simulation.get('msg')
        if msg is not None:
            assert msg.__class__ is suds.byte_str_class
            return self.send(_parse(msg))
        msg = self.method.binding.input.get_message(self.method, args, kwargs)
        log.debug('inject (simulated) send message:\n%s', msg)
        reply = simulation.get('reply')
        if reply is not None:
            assert reply.__class__ is suds.byte_str_class
            status = simulation.get('status')
            description = simulation.get('description')
            if description is None:
                description = 'injected reply'
            return self.process_reply(reply, status, description)
        raise Exception('reply or msg injection parameter expected')