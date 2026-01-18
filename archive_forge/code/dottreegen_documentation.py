from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.tree import CommonTreeAdaptor
from six.moves import range
import stringtemplate3

    Generate DOT (graphviz) for a whole tree not just a node.
    For example, 3+4*5 should generate:

    digraph {
        node [shape=plaintext, fixedsize=true, fontsize=11, fontname="Courier",
            width=.4, height=.2];
        edge [arrowsize=.7]
        "+"->3
        "+"->"*"
        "*"->4
        "*"->5
    }

    Return the ST not a string in case people want to alter.

    Takes a Tree interface object.

    Example of invokation:

        import antlr3
        import antlr3.extras

        input = antlr3.ANTLRInputStream(sys.stdin)
        lex = TLexer(input)
        tokens = antlr3.CommonTokenStream(lex)
        parser = TParser(tokens)
        tree = parser.e().tree
        print tree.toStringTree()
        st = antlr3.extras.toDOT(t)
        print st

    