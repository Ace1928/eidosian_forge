from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
class GetReply(RPCReply):
    """Adds attributes for the *data* element to `RPCReply`."""

    def _parsing_hook(self, root):
        self._data = None
        if not self._errors:
            self._data = root.find(qualify('data'))

    @property
    def data_ele(self):
        """*data* element as an :class:`~xml.etree.ElementTree.Element`"""
        if not self._parsed:
            self.parse()
        return self._data

    @property
    def data_xml(self):
        """*data* element as an XML string"""
        if not self._parsed:
            self.parse()
        return to_xml(self._data)
    data = data_ele
    'Same as :attr:`data_ele`'