import string
from xml.dom import Node
def _do_pi(self, node):
    """_do_pi(self, node) -> None
        Process a PI node. Render a leading or trailing #xA if the
        document order of the PI is greater or lesser (respectively)
        than the document element.
        """
    if not _in_subset(self.subset, node):
        return
    W = self.write
    if self.documentOrder == _GreaterElement:
        W('\n')
    W('<?')
    W(node.nodeName)
    s = node.data
    if s:
        W(' ')
        W(s)
    W('?>')
    if self.documentOrder == _LesserElement:
        W('\n')