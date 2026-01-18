import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
class GEXF:
    versions = {'1.1draft': {'NS_GEXF': 'http://www.gexf.net/1.1draft', 'NS_VIZ': 'http://www.gexf.net/1.1draft/viz', 'NS_XSI': 'http://www.w3.org/2001/XMLSchema-instance', 'SCHEMALOCATION': ' '.join(['http://www.gexf.net/1.1draft', 'http://www.gexf.net/1.1draft/gexf.xsd']), 'VERSION': '1.1'}, '1.2draft': {'NS_GEXF': 'http://www.gexf.net/1.2draft', 'NS_VIZ': 'http://www.gexf.net/1.2draft/viz', 'NS_XSI': 'http://www.w3.org/2001/XMLSchema-instance', 'SCHEMALOCATION': ' '.join(['http://www.gexf.net/1.2draft', 'http://www.gexf.net/1.2draft/gexf.xsd']), 'VERSION': '1.2'}}

    def construct_types(self):
        types = [(int, 'integer'), (float, 'float'), (float, 'double'), (bool, 'boolean'), (list, 'string'), (dict, 'string'), (int, 'long'), (str, 'liststring'), (str, 'anyURI'), (str, 'string')]
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            types = [(np.float64, 'float'), (np.float32, 'float'), (np.float16, 'float'), (np.int_, 'int'), (np.int8, 'int'), (np.int16, 'int'), (np.int32, 'int'), (np.int64, 'int'), (np.uint8, 'int'), (np.uint16, 'int'), (np.uint32, 'int'), (np.uint64, 'int'), (np.int_, 'int'), (np.intc, 'int'), (np.intp, 'int')] + types
        self.xml_type = dict(types)
        self.python_type = dict((reversed(a) for a in types))
    convert_bool = {'true': True, 'false': False, 'True': True, 'False': False, '0': False, 0: False, '1': True, 1: True}

    def set_version(self, version):
        d = self.versions.get(version)
        if d is None:
            raise nx.NetworkXError(f'Unknown GEXF version {version}.')
        self.NS_GEXF = d['NS_GEXF']
        self.NS_VIZ = d['NS_VIZ']
        self.NS_XSI = d['NS_XSI']
        self.SCHEMALOCATION = d['SCHEMALOCATION']
        self.VERSION = d['VERSION']
        self.version = version