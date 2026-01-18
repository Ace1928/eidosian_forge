from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.tree.Tree import ParseTreeListener, ParseTree, TerminalNodeImpl, ErrorNodeImpl, TerminalNode, \
class InterpreterRuleContext(ParserRuleContext):

    def __init__(self, parent: ParserRuleContext, invokingStateNumber: int, ruleIndex: int):
        super().__init__(parent, invokingStateNumber)
        self.ruleIndex = ruleIndex