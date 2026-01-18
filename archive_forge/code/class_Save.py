from lxml import etree
from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
class Save(RPC):

    def request(self, filename=None):
        node = new_ele('save')
        sub_ele(node, 'file').text = filename
        return self._request(node)