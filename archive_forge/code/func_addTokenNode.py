from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.tree.Tree import ParseTreeListener, ParseTree, TerminalNodeImpl, ErrorNodeImpl, TerminalNode, \
def addTokenNode(self, token: Token):
    node = TerminalNodeImpl(token)
    self.addChild(node)
    node.parentCtx = self
    return node