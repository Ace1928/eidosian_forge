from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
def __applytns(self, root):
    """Make sure included schema has the same target namespace."""
    TNS = 'targetNamespace'
    tns = root.get(TNS)
    if tns is None:
        tns = self.schema.tns[1]
        root.set(TNS, tns)
    elif self.schema.tns[1] != tns:
        raise Exception('%s mismatch' % TNS)