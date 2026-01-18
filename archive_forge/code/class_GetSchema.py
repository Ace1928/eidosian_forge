from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
class GetSchema(RPC):
    """The *get-schema* RPC."""
    REPLY_CLS = GetSchemaReply
    'See :class:`GetReply`.'

    def request(self, identifier, version=None, format=None):
        """Retrieve a named schema, with optional revision and type.

        *identifier* name of the schema to be retrieved

        *version* version of schema to get

        *format* format of the schema to be retrieved, yang is the default

        :seealso: :ref:`filter_params`"""
        self._huge_tree = True
        node = etree.Element(qualify('get-schema', NETCONF_MONITORING_NS))
        if identifier is not None:
            elem = etree.Element(qualify('identifier', NETCONF_MONITORING_NS))
            elem.text = identifier
            node.append(elem)
        if version is not None:
            elem = etree.Element(qualify('version', NETCONF_MONITORING_NS))
            elem.text = version
            node.append(elem)
        if format is not None:
            elem = etree.Element(qualify('format', NETCONF_MONITORING_NS))
            elem.text = format
            node.append(elem)
        return self._request(node)