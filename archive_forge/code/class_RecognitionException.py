from antlr3.constants import INVALID_TOKEN_TYPE
class RecognitionException(Exception):
    """@brief The root of the ANTLR exception hierarchy.

    To avoid English-only error messages and to generally make things
    as flexible as possible, these exceptions are not created with strings,
    but rather the information necessary to generate an error.  Then
    the various reporting methods in Parser and Lexer can be overridden
    to generate a localized error message.  For example, MismatchedToken
    exceptions are built with the expected token type.
    So, don't expect getMessage() to return anything.

    Note that as of Java 1.4, you can access the stack trace, which means
    that you can compute the complete trace of rules from the start symbol.
    This gives you considerable context information with which to generate
    useful error messages.

    ANTLR generates code that throws exceptions upon recognition error and
    also generates code to catch these exceptions in each rule.  If you
    want to quit upon first error, you can turn off the automatic error
    handling mechanism using rulecatch action, but you still need to
    override methods mismatch and recoverFromMismatchSet.

    In general, the recognition exceptions can track where in a grammar a
    problem occurred and/or what was the expected input.  While the parser
    knows its state (such as current input symbol and line info) that
    state can change before the exception is reported so current token index
    is computed and stored at exception time.  From this info, you can
    perhaps print an entire line of input not just a single token, for example.
    Better to just say the recognizer had a problem and then let the parser
    figure out a fancy report.

    """

    def __init__(self, input=None):
        Exception.__init__(self)
        self.input = None
        self.index = None
        self.token = None
        self.node = None
        self.c = None
        self.line = None
        self.charPositionInLine = None
        self.approximateLineInfo = False
        if input is not None:
            self.input = input
            self.index = input.index()
            from antlr3.streams import TokenStream, CharStream
            from antlr3.tree import TreeNodeStream
            if isinstance(self.input, TokenStream):
                self.token = self.input.LT(1)
                self.line = self.token.line
                self.charPositionInLine = self.token.charPositionInLine
            if isinstance(self.input, TreeNodeStream):
                self.extractInformationFromTreeNodeStream(self.input)
            elif isinstance(self.input, CharStream):
                self.c = self.input.LT(1)
                self.line = self.input.line
                self.charPositionInLine = self.input.charPositionInLine
            else:
                self.c = self.input.LA(1)

    def extractInformationFromTreeNodeStream(self, nodes):
        from antlr3.tree import Tree, CommonTree
        from antlr3.tokens import CommonToken
        self.node = nodes.LT(1)
        adaptor = nodes.adaptor
        payload = adaptor.getToken(self.node)
        if payload is not None:
            self.token = payload
            if payload.line <= 0:
                i = -1
                priorNode = nodes.LT(i)
                while priorNode is not None:
                    priorPayload = adaptor.getToken(priorNode)
                    if priorPayload is not None and priorPayload.line > 0:
                        self.line = priorPayload.line
                        self.charPositionInLine = priorPayload.charPositionInLine
                        self.approximateLineInfo = True
                        break
                    i -= 1
                    priorNode = nodes.LT(i)
            else:
                self.line = payload.line
                self.charPositionInLine = payload.charPositionInLine
        elif isinstance(self.node, Tree):
            self.line = self.node.line
            self.charPositionInLine = self.node.charPositionInLine
            if isinstance(self.node, CommonTree):
                self.token = self.node.token
        else:
            type = adaptor.getType(self.node)
            text = adaptor.getText(self.node)
            self.token = CommonToken(type=type, text=text)

    def getUnexpectedType(self):
        """Return the token type or char of the unexpected input element"""
        from antlr3.streams import TokenStream
        from antlr3.tree import TreeNodeStream
        if isinstance(self.input, TokenStream):
            return self.token.type
        elif isinstance(self.input, TreeNodeStream):
            adaptor = self.input.treeAdaptor
            return adaptor.getType(self.node)
        else:
            return self.c
    unexpectedType = property(getUnexpectedType)