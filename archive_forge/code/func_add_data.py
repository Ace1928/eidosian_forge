import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def add_data(self, name, element_type, value, scope='all', default=None):
    """
        Make a data element for an edge or a node. Keep a log of the
        type in the keys table.
        """
    if element_type not in self.xml_type:
        raise nx.NetworkXError(f'GraphML writer does not support {element_type} as data values.')
    keyid = self.get_key(name, self.get_xml_type(element_type), scope, default)
    data_element = self.myElement('data', key=keyid)
    data_element.text = str(value)
    return data_element