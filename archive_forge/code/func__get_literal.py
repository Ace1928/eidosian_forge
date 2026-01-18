from functools import reduce
from datetime import datetime
import re
def _get_literal(Pnode):
    """
        Get (recursively) the full text from a DOM Node.
    
        @param Pnode: DOM Node
        @return: string
        """
    rc = ''
    for node in Pnode.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
        elif node.nodeType == node.ELEMENT_NODE:
            rc = rc + _get_literal(node)
    if state.options.space_preserve:
        return rc
    else:
        return re.sub('(\\r| |\\n|\\t)+', ' ', rc).strip()