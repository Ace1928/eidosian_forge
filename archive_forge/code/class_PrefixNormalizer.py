from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
class PrefixNormalizer:
    """
    The prefix normalizer provides namespace prefix normalization.

    @ivar node: A node to normalize.
    @type node: L{Element}
    @ivar branch: The nodes flattened branch.
    @type branch: [L{Element},...]
    @ivar namespaces: A unique list of namespaces (URI).
    @type namespaces: [str,...]
    @ivar prefixes: A reverse dict of prefixes.
    @type prefixes: {u: p}

    """

    @classmethod
    def apply(cls, node):
        """
        Normalize the specified node.

        @param node: A node to normalize.
        @type node: L{Element}
        @return: The normalized node.
        @rtype: L{Element}

        """
        return PrefixNormalizer(node).refit()

    def __init__(self, node):
        """
        @param node: A node to normalize.
        @type node: L{Element}

        """
        self.node = node
        self.branch = node.branch()
        self.namespaces = self.getNamespaces()
        self.prefixes = self.genPrefixes()

    def getNamespaces(self):
        """
        Get the I{unique} set of namespaces referenced in the branch.

        @return: A set of namespaces.
        @rtype: set

        """
        s = set()
        for n in self.branch + self.node.ancestors():
            if self.permit(n.expns):
                s.add(n.expns)
            s = s.union(self.pset(n))
        return s

    def pset(self, n):
        """
        Convert the nodes nsprefixes into a set.

        @param n: A node.
        @type n: L{Element}
        @return: A set of namespaces.
        @rtype: set

        """
        s = set()
        for ns in list(n.nsprefixes.items()):
            if self.permit(ns):
                s.add(ns[1])
        return s

    def genPrefixes(self):
        """
        Generate a I{reverse} mapping of unique prefixes for all namespaces.

        @return: A reverse dict of prefixes.
        @rtype: {u: p}

        """
        prefixes = {}
        n = 0
        for u in self.namespaces:
            prefixes[u] = 'ns%d' % (n,)
            n += 1
        return prefixes

    def refit(self):
        """Refit (normalize) the prefixes in the node."""
        self.refitNodes()
        self.refitMappings()

    def refitNodes(self):
        """Refit (normalize) all of the nodes in the branch."""
        for n in self.branch:
            if n.prefix is not None:
                ns = n.namespace()
                if self.permit(ns):
                    n.prefix = self.prefixes[ns[1]]
            self.refitAttrs(n)

    def refitAttrs(self, n):
        """
        Refit (normalize) all of the attributes in the node.

        @param n: A node.
        @type n: L{Element}

        """
        for a in n.attributes:
            self.refitAddr(a)

    def refitAddr(self, a):
        """
        Refit (normalize) the attribute.

        @param a: An attribute.
        @type a: L{Attribute}

        """
        if a.prefix is not None:
            ns = a.namespace()
            if self.permit(ns):
                a.prefix = self.prefixes[ns[1]]
        self.refitValue(a)

    def refitValue(self, a):
        """
        Refit (normalize) the attribute's value.

        @param a: An attribute.
        @type a: L{Attribute}

        """
        p, name = splitPrefix(a.getValue())
        if p is None:
            return
        ns = a.resolvePrefix(p)
        if self.permit(ns):
            p = self.prefixes[ns[1]]
            a.setValue(':'.join((p, name)))

    def refitMappings(self):
        """Refit (normalize) all of the nsprefix mappings."""
        for n in self.branch:
            n.nsprefixes = {}
        n = self.node
        for u, p in list(self.prefixes.items()):
            n.addPrefix(p, u)

    def permit(self, ns):
        """
        Get whether the I{ns} is to be normalized.

        @param ns: A namespace.
        @type ns: (p, u)
        @return: True if to be included.
        @rtype: boolean

        """
        return not self.skip(ns)

    def skip(self, ns):
        """
        Get whether the I{ns} is to B{not} be normalized.

        @param ns: A namespace.
        @type ns: (p, u)
        @return: True if to be skipped.
        @rtype: boolean

        """
        return ns is None or ns in (Namespace.default, Namespace.xsdns, Namespace.xsins, Namespace.xmlns)