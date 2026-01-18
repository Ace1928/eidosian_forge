import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class GraphML:
    NS_GRAPHML = 'http://graphml.graphdrawing.org/xmlns'
    NS_XSI = 'http://www.w3.org/2001/XMLSchema-instance'
    NS_Y = 'http://www.yworks.com/xml/graphml'
    SCHEMALOCATION = ' '.join(['http://graphml.graphdrawing.org/xmlns', 'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'])

    def construct_types(self):
        types = [(int, 'integer'), (str, 'yfiles'), (str, 'string'), (int, 'int'), (int, 'long'), (float, 'float'), (float, 'double'), (bool, 'boolean')]
        try:
            import numpy as np
        except:
            pass
        else:
            types = [(np.float64, 'float'), (np.float32, 'float'), (np.float16, 'float'), (np.int_, 'int'), (np.int8, 'int'), (np.int16, 'int'), (np.int32, 'int'), (np.int64, 'int'), (np.uint8, 'int'), (np.uint16, 'int'), (np.uint32, 'int'), (np.uint64, 'int'), (np.int_, 'int'), (np.intc, 'int'), (np.intp, 'int')] + types
        self.xml_type = dict(types)
        self.python_type = dict((reversed(a) for a in types))
    convert_bool = {'true': True, 'false': False, '0': False, 0: False, '1': True, 1: True}

    def get_xml_type(self, key):
        """Wrapper around the xml_type dict that raises a more informative
        exception message when a user attempts to use data of a type not
        supported by GraphML."""
        try:
            return self.xml_type[key]
        except KeyError as err:
            raise TypeError(f'GraphML does not support type {type(key)} as data values.') from err