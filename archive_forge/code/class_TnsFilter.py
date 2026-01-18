from suds.sax import Namespace
from suds.sax.element import Element
from suds.plugin import DocumentPlugin, DocumentContext
from logging import getLogger
class TnsFilter:
    """
    Target Namespace filter.
    @ivar tns: A list of target namespaces.
    @type tns: [str,...]
    """

    def __init__(self, *tns):
        """
        @param tns: A list of target namespaces.
        @type tns: [str,...]
        """
        self.tns = []
        self.add(*tns)

    def add(self, *tns):
        """
        Add I{targetNamespaces} to be added.
        @param tns: A list of target namespaces.
        @type tns: [str,...]
        """
        self.tns += tns

    def match(self, root, ns):
        """
        Match by I{targetNamespace} excluding those that
        are equal to the specified namespace to prevent
        adding an import to itself.
        @param root: A schema root.
        @type root: L{Element}
        """
        tns = root.get('targetNamespace')
        if len(self.tns):
            matched = tns in self.tns
        else:
            matched = 1
        itself = ns == tns
        return matched and (not itself)