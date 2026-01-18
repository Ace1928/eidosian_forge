from antlr4 import *
from io import StringIO
import sys
class AutolevParser(Parser):
    grammarFileName = 'Autolev.g4'
    atn = ATNDeserializer().deserialize(serializedATN())
    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]
    sharedContextCache = PredictionContextCache()
    literalNames = ['<INVALID>', "'['", "']'", "'='", "'+='", "'-='", "':='", "'*='", "'/='", "'^='", "','", "'''", "'('", "')'", "'{'", "'}'", "':'", "'+'", "'-'", "';'", "'.'", "'>'", "'0>'", "'1>>'", "'^'", "'*'", "'/'"]
    symbolicNames = ['<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', 'Mass', 'Inertia', 'Input', 'Output', 'Save', 'UnitSystem', 'Encode', 'Newtonian', 'Frames', 'Bodies', 'Particles', 'Points', 'Constants', 'Specifieds', 'Imaginary', 'Variables', 'MotionVariables', 'INT', 'FLOAT', 'EXP', 'LINE_COMMENT', 'ID', 'WS']
    RULE_prog = 0
    RULE_stat = 1
    RULE_assignment = 2
    RULE_equals = 3
    RULE_index = 4
    RULE_diff = 5
    RULE_functionCall = 6
    RULE_varDecl = 7
    RULE_varType = 8
    RULE_varDecl2 = 9
    RULE_ranges = 10
    RULE_massDecl = 11
    RULE_massDecl2 = 12
    RULE_inertiaDecl = 13
    RULE_matrix = 14
    RULE_matrixInOutput = 15
    RULE_codeCommands = 16
    RULE_settings = 17
    RULE_units = 18
    RULE_inputs = 19
    RULE_id_diff = 20
    RULE_inputs2 = 21
    RULE_outputs = 22
    RULE_outputs2 = 23
    RULE_codegen = 24
    RULE_commands = 25
    RULE_vec = 26
    RULE_expr = 27
    ruleNames = ['prog', 'stat', 'assignment', 'equals', 'index', 'diff', 'functionCall', 'varDecl', 'varType', 'varDecl2', 'ranges', 'massDecl', 'massDecl2', 'inertiaDecl', 'matrix', 'matrixInOutput', 'codeCommands', 'settings', 'units', 'inputs', 'id_diff', 'inputs2', 'outputs', 'outputs2', 'codegen', 'commands', 'vec', 'expr']
    EOF = Token.EOF
    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    T__16 = 17
    T__17 = 18
    T__18 = 19
    T__19 = 20
    T__20 = 21
    T__21 = 22
    T__22 = 23
    T__23 = 24
    T__24 = 25
    T__25 = 26
    Mass = 27
    Inertia = 28
    Input = 29
    Output = 30
    Save = 31
    UnitSystem = 32
    Encode = 33
    Newtonian = 34
    Frames = 35
    Bodies = 36
    Particles = 37
    Points = 38
    Constants = 39
    Specifieds = 40
    Imaginary = 41
    Variables = 42
    MotionVariables = 43
    INT = 44
    FLOAT = 45
    EXP = 46
    LINE_COMMENT = 47
    ID = 48
    WS = 49

    def __init__(self, input: TokenStream, output: TextIO=sys.stdout):
        super().__init__(input, output)
        self.checkVersion('4.11.1')
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None

    class ProgContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def stat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.StatContext)
            else:
                return self.getTypedRuleContext(AutolevParser.StatContext, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_prog

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterProg'):
                listener.enterProg(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitProg'):
                listener.exitProg(self)

    def prog(self):
        localctx = AutolevParser.ProgContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_prog)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 57
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 56
                self.stat()
                self.state = 59
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la & ~63 == 0 and 1 << _la & 299067041120256 != 0):
                    break
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class StatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def varDecl(self):
            return self.getTypedRuleContext(AutolevParser.VarDeclContext, 0)

        def functionCall(self):
            return self.getTypedRuleContext(AutolevParser.FunctionCallContext, 0)

        def codeCommands(self):
            return self.getTypedRuleContext(AutolevParser.CodeCommandsContext, 0)

        def massDecl(self):
            return self.getTypedRuleContext(AutolevParser.MassDeclContext, 0)

        def inertiaDecl(self):
            return self.getTypedRuleContext(AutolevParser.InertiaDeclContext, 0)

        def assignment(self):
            return self.getTypedRuleContext(AutolevParser.AssignmentContext, 0)

        def settings(self):
            return self.getTypedRuleContext(AutolevParser.SettingsContext, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_stat

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterStat'):
                listener.enterStat(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitStat'):
                listener.exitStat(self)

    def stat(self):
        localctx = AutolevParser.StatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_stat)
        try:
            self.state = 68
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 1, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 61
                self.varDecl()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 62
                self.functionCall()
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 63
                self.codeCommands()
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 64
                self.massDecl()
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 65
                self.inertiaDecl()
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 66
                self.assignment()
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 67
                self.settings()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return AutolevParser.RULE_assignment

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class VecAssignContext(AssignmentContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def vec(self):
            return self.getTypedRuleContext(AutolevParser.VecContext, 0)

        def equals(self):
            return self.getTypedRuleContext(AutolevParser.EqualsContext, 0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterVecAssign'):
                listener.enterVecAssign(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitVecAssign'):
                listener.exitVecAssign(self)

    class RegularAssignContext(AssignmentContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def equals(self):
            return self.getTypedRuleContext(AutolevParser.EqualsContext, 0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

        def diff(self):
            return self.getTypedRuleContext(AutolevParser.DiffContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterRegularAssign'):
                listener.enterRegularAssign(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitRegularAssign'):
                listener.exitRegularAssign(self)

    class IndexAssignContext(AssignmentContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def index(self):
            return self.getTypedRuleContext(AutolevParser.IndexContext, 0)

        def equals(self):
            return self.getTypedRuleContext(AutolevParser.EqualsContext, 0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterIndexAssign'):
                listener.enterIndexAssign(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitIndexAssign'):
                listener.exitIndexAssign(self)

    def assignment(self):
        localctx = AutolevParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_assignment)
        self._la = 0
        try:
            self.state = 88
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 3, self._ctx)
            if la_ == 1:
                localctx = AutolevParser.VecAssignContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 70
                self.vec()
                self.state = 71
                self.equals()
                self.state = 72
                self.expr(0)
                pass
            elif la_ == 2:
                localctx = AutolevParser.IndexAssignContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 74
                self.match(AutolevParser.ID)
                self.state = 75
                self.match(AutolevParser.T__0)
                self.state = 76
                self.index()
                self.state = 77
                self.match(AutolevParser.T__1)
                self.state = 78
                self.equals()
                self.state = 79
                self.expr(0)
                pass
            elif la_ == 3:
                localctx = AutolevParser.RegularAssignContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 81
                self.match(AutolevParser.ID)
                self.state = 83
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 11:
                    self.state = 82
                    self.diff()
                self.state = 85
                self.equals()
                self.state = 86
                self.expr(0)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class EqualsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return AutolevParser.RULE_equals

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterEquals'):
                listener.enterEquals(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitEquals'):
                listener.exitEquals(self)

    def equals(self):
        localctx = AutolevParser.EqualsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_equals)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 90
            _la = self._input.LA(1)
            if not (_la & ~63 == 0 and 1 << _la & 1016 != 0):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IndexContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_index

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterIndex'):
                listener.enterIndex(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitIndex'):
                listener.exitIndex(self)

    def index(self):
        localctx = AutolevParser.IndexContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_index)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 92
            self.expr(0)
            self.state = 97
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10:
                self.state = 93
                self.match(AutolevParser.T__9)
                self.state = 94
                self.expr(0)
                self.state = 99
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DiffContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return AutolevParser.RULE_diff

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterDiff'):
                listener.enterDiff(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitDiff'):
                listener.exitDiff(self)

    def diff(self):
        localctx = AutolevParser.DiffContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_diff)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 101
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 100
                self.match(AutolevParser.T__10)
                self.state = 103
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not _la == 11:
                    break
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FunctionCallContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def Mass(self):
            return self.getToken(AutolevParser.Mass, 0)

        def Inertia(self):
            return self.getToken(AutolevParser.Inertia, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_functionCall

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterFunctionCall'):
                listener.enterFunctionCall(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitFunctionCall'):
                listener.exitFunctionCall(self)

    def functionCall(self):
        localctx = AutolevParser.FunctionCallContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_functionCall)
        self._la = 0
        try:
            self.state = 131
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [48]:
                self.enterOuterAlt(localctx, 1)
                self.state = 105
                self.match(AutolevParser.ID)
                self.state = 106
                self.match(AutolevParser.T__11)
                self.state = 115
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 404620694540290 != 0:
                    self.state = 107
                    self.expr(0)
                    self.state = 112
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 10:
                        self.state = 108
                        self.match(AutolevParser.T__9)
                        self.state = 109
                        self.expr(0)
                        self.state = 114
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 117
                self.match(AutolevParser.T__12)
                pass
            elif token in [27, 28]:
                self.enterOuterAlt(localctx, 2)
                self.state = 118
                _la = self._input.LA(1)
                if not (_la == 27 or _la == 28):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 119
                self.match(AutolevParser.T__11)
                self.state = 128
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 48:
                    self.state = 120
                    self.match(AutolevParser.ID)
                    self.state = 125
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 10:
                        self.state = 121
                        self.match(AutolevParser.T__9)
                        self.state = 122
                        self.match(AutolevParser.ID)
                        self.state = 127
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 130
                self.match(AutolevParser.T__12)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class VarDeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def varType(self):
            return self.getTypedRuleContext(AutolevParser.VarTypeContext, 0)

        def varDecl2(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.VarDecl2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.VarDecl2Context, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_varDecl

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterVarDecl'):
                listener.enterVarDecl(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitVarDecl'):
                listener.exitVarDecl(self)

    def varDecl(self):
        localctx = AutolevParser.VarDeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_varDecl)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 133
            self.varType()
            self.state = 134
            self.varDecl2()
            self.state = 139
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10:
                self.state = 135
                self.match(AutolevParser.T__9)
                self.state = 136
                self.varDecl2()
                self.state = 141
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class VarTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Newtonian(self):
            return self.getToken(AutolevParser.Newtonian, 0)

        def Frames(self):
            return self.getToken(AutolevParser.Frames, 0)

        def Bodies(self):
            return self.getToken(AutolevParser.Bodies, 0)

        def Particles(self):
            return self.getToken(AutolevParser.Particles, 0)

        def Points(self):
            return self.getToken(AutolevParser.Points, 0)

        def Constants(self):
            return self.getToken(AutolevParser.Constants, 0)

        def Specifieds(self):
            return self.getToken(AutolevParser.Specifieds, 0)

        def Imaginary(self):
            return self.getToken(AutolevParser.Imaginary, 0)

        def Variables(self):
            return self.getToken(AutolevParser.Variables, 0)

        def MotionVariables(self):
            return self.getToken(AutolevParser.MotionVariables, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_varType

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterVarType'):
                listener.enterVarType(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitVarType'):
                listener.exitVarType(self)

    def varType(self):
        localctx = AutolevParser.VarTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_varType)
        self._la = 0
        try:
            self.state = 164
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [34]:
                self.enterOuterAlt(localctx, 1)
                self.state = 142
                self.match(AutolevParser.Newtonian)
                pass
            elif token in [35]:
                self.enterOuterAlt(localctx, 2)
                self.state = 143
                self.match(AutolevParser.Frames)
                pass
            elif token in [36]:
                self.enterOuterAlt(localctx, 3)
                self.state = 144
                self.match(AutolevParser.Bodies)
                pass
            elif token in [37]:
                self.enterOuterAlt(localctx, 4)
                self.state = 145
                self.match(AutolevParser.Particles)
                pass
            elif token in [38]:
                self.enterOuterAlt(localctx, 5)
                self.state = 146
                self.match(AutolevParser.Points)
                pass
            elif token in [39]:
                self.enterOuterAlt(localctx, 6)
                self.state = 147
                self.match(AutolevParser.Constants)
                pass
            elif token in [40]:
                self.enterOuterAlt(localctx, 7)
                self.state = 148
                self.match(AutolevParser.Specifieds)
                pass
            elif token in [41]:
                self.enterOuterAlt(localctx, 8)
                self.state = 149
                self.match(AutolevParser.Imaginary)
                pass
            elif token in [42]:
                self.enterOuterAlt(localctx, 9)
                self.state = 150
                self.match(AutolevParser.Variables)
                self.state = 154
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 11:
                    self.state = 151
                    self.match(AutolevParser.T__10)
                    self.state = 156
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            elif token in [43]:
                self.enterOuterAlt(localctx, 10)
                self.state = 157
                self.match(AutolevParser.MotionVariables)
                self.state = 161
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 11:
                    self.state = 158
                    self.match(AutolevParser.T__10)
                    self.state = 163
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class VarDecl2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def INT(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.INT)
            else:
                return self.getToken(AutolevParser.INT, i)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_varDecl2

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterVarDecl2'):
                listener.enterVarDecl2(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitVarDecl2'):
                listener.exitVarDecl2(self)

    def varDecl2(self):
        localctx = AutolevParser.VarDecl2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_varDecl2)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 166
            self.match(AutolevParser.ID)
            self.state = 172
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 15, self._ctx)
            if la_ == 1:
                self.state = 167
                self.match(AutolevParser.T__13)
                self.state = 168
                self.match(AutolevParser.INT)
                self.state = 169
                self.match(AutolevParser.T__9)
                self.state = 170
                self.match(AutolevParser.INT)
                self.state = 171
                self.match(AutolevParser.T__14)
            self.state = 188
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 17, self._ctx)
            if la_ == 1:
                self.state = 174
                self.match(AutolevParser.T__13)
                self.state = 175
                self.match(AutolevParser.INT)
                self.state = 176
                self.match(AutolevParser.T__15)
                self.state = 177
                self.match(AutolevParser.INT)
                self.state = 184
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 10:
                    self.state = 178
                    self.match(AutolevParser.T__9)
                    self.state = 179
                    self.match(AutolevParser.INT)
                    self.state = 180
                    self.match(AutolevParser.T__15)
                    self.state = 181
                    self.match(AutolevParser.INT)
                    self.state = 186
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 187
                self.match(AutolevParser.T__14)
            self.state = 193
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 14:
                self.state = 190
                self.match(AutolevParser.T__13)
                self.state = 191
                self.match(AutolevParser.INT)
                self.state = 192
                self.match(AutolevParser.T__14)
            self.state = 196
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 17 or _la == 18:
                self.state = 195
                _la = self._input.LA(1)
                if not (_la == 17 or _la == 18):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
            self.state = 201
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 11:
                self.state = 198
                self.match(AutolevParser.T__10)
                self.state = 203
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 206
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 3:
                self.state = 204
                self.match(AutolevParser.T__2)
                self.state = 205
                self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class RangesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.INT)
            else:
                return self.getToken(AutolevParser.INT, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_ranges

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterRanges'):
                listener.enterRanges(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitRanges'):
                listener.exitRanges(self)

    def ranges(self):
        localctx = AutolevParser.RangesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_ranges)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 208
            self.match(AutolevParser.T__13)
            self.state = 209
            self.match(AutolevParser.INT)
            self.state = 210
            self.match(AutolevParser.T__15)
            self.state = 211
            self.match(AutolevParser.INT)
            self.state = 218
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10:
                self.state = 212
                self.match(AutolevParser.T__9)
                self.state = 213
                self.match(AutolevParser.INT)
                self.state = 214
                self.match(AutolevParser.T__15)
                self.state = 215
                self.match(AutolevParser.INT)
                self.state = 220
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 221
            self.match(AutolevParser.T__14)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MassDeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Mass(self):
            return self.getToken(AutolevParser.Mass, 0)

        def massDecl2(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.MassDecl2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.MassDecl2Context, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_massDecl

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterMassDecl'):
                listener.enterMassDecl(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitMassDecl'):
                listener.exitMassDecl(self)

    def massDecl(self):
        localctx = AutolevParser.MassDeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_massDecl)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 223
            self.match(AutolevParser.Mass)
            self.state = 224
            self.massDecl2()
            self.state = 229
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10:
                self.state = 225
                self.match(AutolevParser.T__9)
                self.state = 226
                self.massDecl2()
                self.state = 231
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MassDecl2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_massDecl2

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterMassDecl2'):
                listener.enterMassDecl2(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitMassDecl2'):
                listener.exitMassDecl2(self)

    def massDecl2(self):
        localctx = AutolevParser.MassDecl2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_massDecl2)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 232
            self.match(AutolevParser.ID)
            self.state = 233
            self.match(AutolevParser.T__2)
            self.state = 234
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class InertiaDeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Inertia(self):
            return self.getToken(AutolevParser.Inertia, 0)

        def ID(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_inertiaDecl

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterInertiaDecl'):
                listener.enterInertiaDecl(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitInertiaDecl'):
                listener.exitInertiaDecl(self)

    def inertiaDecl(self):
        localctx = AutolevParser.InertiaDeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_inertiaDecl)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 236
            self.match(AutolevParser.Inertia)
            self.state = 237
            self.match(AutolevParser.ID)
            self.state = 241
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 12:
                self.state = 238
                self.match(AutolevParser.T__11)
                self.state = 239
                self.match(AutolevParser.ID)
                self.state = 240
                self.match(AutolevParser.T__12)
            self.state = 245
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 243
                self.match(AutolevParser.T__9)
                self.state = 244
                self.expr(0)
                self.state = 247
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not _la == 10:
                    break
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MatrixContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_matrix

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterMatrix'):
                listener.enterMatrix(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitMatrix'):
                listener.exitMatrix(self)

    def matrix(self):
        localctx = AutolevParser.MatrixContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_matrix)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 249
            self.match(AutolevParser.T__0)
            self.state = 250
            self.expr(0)
            self.state = 255
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10 or _la == 19:
                self.state = 251
                _la = self._input.LA(1)
                if not (_la == 10 or _la == 19):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 252
                self.expr(0)
                self.state = 257
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 258
            self.match(AutolevParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MatrixInOutputContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def FLOAT(self):
            return self.getToken(AutolevParser.FLOAT, 0)

        def INT(self):
            return self.getToken(AutolevParser.INT, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_matrixInOutput

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterMatrixInOutput'):
                listener.enterMatrixInOutput(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitMatrixInOutput'):
                listener.exitMatrixInOutput(self)

    def matrixInOutput(self):
        localctx = AutolevParser.MatrixInOutputContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_matrixInOutput)
        self._la = 0
        try:
            self.state = 268
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [48]:
                self.enterOuterAlt(localctx, 1)
                self.state = 260
                self.match(AutolevParser.ID)
                self.state = 261
                self.match(AutolevParser.ID)
                self.state = 262
                self.match(AutolevParser.T__2)
                self.state = 264
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 44 or _la == 45:
                    self.state = 263
                    _la = self._input.LA(1)
                    if not (_la == 44 or _la == 45):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                pass
            elif token in [45]:
                self.enterOuterAlt(localctx, 2)
                self.state = 266
                self.match(AutolevParser.FLOAT)
                pass
            elif token in [44]:
                self.enterOuterAlt(localctx, 3)
                self.state = 267
                self.match(AutolevParser.INT)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CodeCommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def units(self):
            return self.getTypedRuleContext(AutolevParser.UnitsContext, 0)

        def inputs(self):
            return self.getTypedRuleContext(AutolevParser.InputsContext, 0)

        def outputs(self):
            return self.getTypedRuleContext(AutolevParser.OutputsContext, 0)

        def codegen(self):
            return self.getTypedRuleContext(AutolevParser.CodegenContext, 0)

        def commands(self):
            return self.getTypedRuleContext(AutolevParser.CommandsContext, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_codeCommands

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterCodeCommands'):
                listener.enterCodeCommands(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitCodeCommands'):
                listener.exitCodeCommands(self)

    def codeCommands(self):
        localctx = AutolevParser.CodeCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_codeCommands)
        try:
            self.state = 275
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [32]:
                self.enterOuterAlt(localctx, 1)
                self.state = 270
                self.units()
                pass
            elif token in [29]:
                self.enterOuterAlt(localctx, 2)
                self.state = 271
                self.inputs()
                pass
            elif token in [30]:
                self.enterOuterAlt(localctx, 3)
                self.state = 272
                self.outputs()
                pass
            elif token in [48]:
                self.enterOuterAlt(localctx, 4)
                self.state = 273
                self.codegen()
                pass
            elif token in [31, 33]:
                self.enterOuterAlt(localctx, 5)
                self.state = 274
                self.commands()
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SettingsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def EXP(self):
            return self.getToken(AutolevParser.EXP, 0)

        def FLOAT(self):
            return self.getToken(AutolevParser.FLOAT, 0)

        def INT(self):
            return self.getToken(AutolevParser.INT, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_settings

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterSettings'):
                listener.enterSettings(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitSettings'):
                listener.exitSettings(self)

    def settings(self):
        localctx = AutolevParser.SettingsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_settings)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 277
            self.match(AutolevParser.ID)
            self.state = 279
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 30, self._ctx)
            if la_ == 1:
                self.state = 278
                _la = self._input.LA(1)
                if not (_la & ~63 == 0 and 1 << _la & 404620279021568 != 0):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class UnitsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def UnitSystem(self):
            return self.getToken(AutolevParser.UnitSystem, 0)

        def ID(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_units

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterUnits'):
                listener.enterUnits(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitUnits'):
                listener.exitUnits(self)

    def units(self):
        localctx = AutolevParser.UnitsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_units)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 281
            self.match(AutolevParser.UnitSystem)
            self.state = 282
            self.match(AutolevParser.ID)
            self.state = 287
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10:
                self.state = 283
                self.match(AutolevParser.T__9)
                self.state = 284
                self.match(AutolevParser.ID)
                self.state = 289
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class InputsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Input(self):
            return self.getToken(AutolevParser.Input, 0)

        def inputs2(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.Inputs2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.Inputs2Context, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_inputs

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterInputs'):
                listener.enterInputs(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitInputs'):
                listener.exitInputs(self)

    def inputs(self):
        localctx = AutolevParser.InputsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_inputs)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 290
            self.match(AutolevParser.Input)
            self.state = 291
            self.inputs2()
            self.state = 296
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10:
                self.state = 292
                self.match(AutolevParser.T__9)
                self.state = 293
                self.inputs2()
                self.state = 298
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Id_diffContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def diff(self):
            return self.getTypedRuleContext(AutolevParser.DiffContext, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_id_diff

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterId_diff'):
                listener.enterId_diff(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitId_diff'):
                listener.exitId_diff(self)

    def id_diff(self):
        localctx = AutolevParser.Id_diffContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_id_diff)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 299
            self.match(AutolevParser.ID)
            self.state = 301
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 11:
                self.state = 300
                self.diff()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Inputs2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def id_diff(self):
            return self.getTypedRuleContext(AutolevParser.Id_diffContext, 0)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_inputs2

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterInputs2'):
                listener.enterInputs2(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitInputs2'):
                listener.exitInputs2(self)

    def inputs2(self):
        localctx = AutolevParser.Inputs2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_inputs2)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 303
            self.id_diff()
            self.state = 304
            self.match(AutolevParser.T__2)
            self.state = 305
            self.expr(0)
            self.state = 307
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 34, self._ctx)
            if la_ == 1:
                self.state = 306
                self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class OutputsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Output(self):
            return self.getToken(AutolevParser.Output, 0)

        def outputs2(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.Outputs2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.Outputs2Context, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_outputs

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterOutputs'):
                listener.enterOutputs(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitOutputs'):
                listener.exitOutputs(self)

    def outputs(self):
        localctx = AutolevParser.OutputsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_outputs)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 309
            self.match(AutolevParser.Output)
            self.state = 310
            self.outputs2()
            self.state = 315
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 10:
                self.state = 311
                self.match(AutolevParser.T__9)
                self.state = 312
                self.outputs2()
                self.state = 317
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Outputs2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_outputs2

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterOutputs2'):
                listener.enterOutputs2(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitOutputs2'):
                listener.exitOutputs2(self)

    def outputs2(self):
        localctx = AutolevParser.Outputs2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_outputs2)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 318
            self.expr(0)
            self.state = 320
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 36, self._ctx)
            if la_ == 1:
                self.state = 319
                self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CodegenContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def functionCall(self):
            return self.getTypedRuleContext(AutolevParser.FunctionCallContext, 0)

        def matrixInOutput(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.MatrixInOutputContext)
            else:
                return self.getTypedRuleContext(AutolevParser.MatrixInOutputContext, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_codegen

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterCodegen'):
                listener.enterCodegen(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitCodegen'):
                listener.exitCodegen(self)

    def codegen(self):
        localctx = AutolevParser.CodegenContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_codegen)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 322
            self.match(AutolevParser.ID)
            self.state = 323
            self.functionCall()
            self.state = 335
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 1:
                self.state = 324
                self.match(AutolevParser.T__0)
                self.state = 325
                self.matrixInOutput()
                self.state = 330
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 10:
                    self.state = 326
                    self.match(AutolevParser.T__9)
                    self.state = 327
                    self.matrixInOutput()
                    self.state = 332
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 333
                self.match(AutolevParser.T__1)
            self.state = 337
            self.match(AutolevParser.ID)
            self.state = 338
            self.match(AutolevParser.T__19)
            self.state = 339
            self.match(AutolevParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Save(self):
            return self.getToken(AutolevParser.Save, 0)

        def ID(self, i: int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def Encode(self):
            return self.getToken(AutolevParser.Encode, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_commands

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterCommands'):
                listener.enterCommands(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitCommands'):
                listener.exitCommands(self)

    def commands(self):
        localctx = AutolevParser.CommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_commands)
        self._la = 0
        try:
            self.state = 354
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [31]:
                self.enterOuterAlt(localctx, 1)
                self.state = 341
                self.match(AutolevParser.Save)
                self.state = 342
                self.match(AutolevParser.ID)
                self.state = 343
                self.match(AutolevParser.T__19)
                self.state = 344
                self.match(AutolevParser.ID)
                pass
            elif token in [33]:
                self.enterOuterAlt(localctx, 2)
                self.state = 345
                self.match(AutolevParser.Encode)
                self.state = 346
                self.match(AutolevParser.ID)
                self.state = 351
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 10:
                    self.state = 347
                    self.match(AutolevParser.T__9)
                    self.state = 348
                    self.match(AutolevParser.ID)
                    self.state = 353
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class VecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_vec

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterVec'):
                listener.enterVec(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitVec'):
                listener.exitVec(self)

    def vec(self):
        localctx = AutolevParser.VecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_vec)
        try:
            self.state = 364
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [48]:
                self.enterOuterAlt(localctx, 1)
                self.state = 356
                self.match(AutolevParser.ID)
                self.state = 358
                self._errHandler.sync(self)
                _alt = 1
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 357
                        self.match(AutolevParser.T__20)
                    else:
                        raise NoViableAltException(self)
                    self.state = 360
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 41, self._ctx)
                pass
            elif token in [22]:
                self.enterOuterAlt(localctx, 2)
                self.state = 362
                self.match(AutolevParser.T__21)
                pass
            elif token in [23]:
                self.enterOuterAlt(localctx, 3)
                self.state = 363
                self.match(AutolevParser.T__22)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return AutolevParser.RULE_expr

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ParensContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterParens'):
                listener.enterParens(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitParens'):
                listener.exitParens(self)

    class VectorOrDyadicContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def vec(self):
            return self.getTypedRuleContext(AutolevParser.VecContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterVectorOrDyadic'):
                listener.enterVectorOrDyadic(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitVectorOrDyadic'):
                listener.exitVectorOrDyadic(self)

    class ExponentContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterExponent'):
                listener.enterExponent(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitExponent'):
                listener.exitExponent(self)

    class MulDivContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterMulDiv'):
                listener.enterMulDiv(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitMulDiv'):
                listener.exitMulDiv(self)

    class AddSubContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterAddSub'):
                listener.enterAddSub(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitAddSub'):
                listener.exitAddSub(self)

    class FloatContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def FLOAT(self):
            return self.getToken(AutolevParser.FLOAT, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterFloat'):
                listener.enterFloat(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitFloat'):
                listener.exitFloat(self)

    class IntContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def INT(self):
            return self.getToken(AutolevParser.INT, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterInt'):
                listener.enterInt(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitInt'):
                listener.exitInt(self)

    class IdEqualsExprContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterIdEqualsExpr'):
                listener.enterIdEqualsExpr(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitIdEqualsExpr'):
                listener.exitIdEqualsExpr(self)

    class NegativeOneContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterNegativeOne'):
                listener.enterNegativeOne(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitNegativeOne'):
                listener.exitNegativeOne(self)

    class FunctionContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def functionCall(self):
            return self.getTypedRuleContext(AutolevParser.FunctionCallContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterFunction'):
                listener.enterFunction(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitFunction'):
                listener.exitFunction(self)

    class RangessContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ranges(self):
            return self.getTypedRuleContext(AutolevParser.RangesContext, 0)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterRangess'):
                listener.enterRangess(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitRangess'):
                listener.exitRangess(self)

    class ColonContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterColon'):
                listener.enterColon(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitColon'):
                listener.exitColon(self)

    class IdContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterId'):
                listener.enterId(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitId'):
                listener.exitId(self)

    class ExpContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def EXP(self):
            return self.getToken(AutolevParser.EXP, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterExp'):
                listener.enterExp(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitExp'):
                listener.exitExp(self)

    class MatricesContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def matrix(self):
            return self.getTypedRuleContext(AutolevParser.MatrixContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterMatrices'):
                listener.enterMatrices(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitMatrices'):
                listener.exitMatrices(self)

    class IndexingContext(ExprContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def expr(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext, i)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'enterIndexing'):
                listener.enterIndexing(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, 'exitIndexing'):
                listener.exitIndexing(self)

    def expr(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AutolevParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 54
        self.enterRecursionRule(localctx, 54, self.RULE_expr, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 408
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 47, self._ctx)
            if la_ == 1:
                localctx = AutolevParser.ExpContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 367
                self.match(AutolevParser.EXP)
                pass
            elif la_ == 2:
                localctx = AutolevParser.NegativeOneContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 368
                self.match(AutolevParser.T__17)
                self.state = 369
                self.expr(12)
                pass
            elif la_ == 3:
                localctx = AutolevParser.FloatContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 370
                self.match(AutolevParser.FLOAT)
                pass
            elif la_ == 4:
                localctx = AutolevParser.IntContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 371
                self.match(AutolevParser.INT)
                pass
            elif la_ == 5:
                localctx = AutolevParser.IdContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 372
                self.match(AutolevParser.ID)
                self.state = 376
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 43, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 373
                        self.match(AutolevParser.T__10)
                    self.state = 378
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 43, self._ctx)
                pass
            elif la_ == 6:
                localctx = AutolevParser.VectorOrDyadicContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 379
                self.vec()
                pass
            elif la_ == 7:
                localctx = AutolevParser.IndexingContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 380
                self.match(AutolevParser.ID)
                self.state = 381
                self.match(AutolevParser.T__0)
                self.state = 382
                self.expr(0)
                self.state = 387
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 10:
                    self.state = 383
                    self.match(AutolevParser.T__9)
                    self.state = 384
                    self.expr(0)
                    self.state = 389
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 390
                self.match(AutolevParser.T__1)
                pass
            elif la_ == 8:
                localctx = AutolevParser.FunctionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 392
                self.functionCall()
                pass
            elif la_ == 9:
                localctx = AutolevParser.MatricesContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 393
                self.matrix()
                pass
            elif la_ == 10:
                localctx = AutolevParser.ParensContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 394
                self.match(AutolevParser.T__11)
                self.state = 395
                self.expr(0)
                self.state = 396
                self.match(AutolevParser.T__12)
                pass
            elif la_ == 11:
                localctx = AutolevParser.RangessContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 399
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 48:
                    self.state = 398
                    self.match(AutolevParser.ID)
                self.state = 401
                self.ranges()
                self.state = 405
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 46, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 402
                        self.match(AutolevParser.T__10)
                    self.state = 407
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 46, self._ctx)
                pass
            self._ctx.stop = self._input.LT(-1)
            self.state = 427
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 49, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 425
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 48, self._ctx)
                    if la_ == 1:
                        localctx = AutolevParser.ExponentContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 410
                        if not self.precpred(self._ctx, 16):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 16)')
                        self.state = 411
                        self.match(AutolevParser.T__23)
                        self.state = 412
                        self.expr(17)
                        pass
                    elif la_ == 2:
                        localctx = AutolevParser.MulDivContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 413
                        if not self.precpred(self._ctx, 15):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 15)')
                        self.state = 414
                        _la = self._input.LA(1)
                        if not (_la == 25 or _la == 26):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 415
                        self.expr(16)
                        pass
                    elif la_ == 3:
                        localctx = AutolevParser.AddSubContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 416
                        if not self.precpred(self._ctx, 14):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 14)')
                        self.state = 417
                        _la = self._input.LA(1)
                        if not (_la == 17 or _la == 18):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 418
                        self.expr(15)
                        pass
                    elif la_ == 4:
                        localctx = AutolevParser.IdEqualsExprContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 419
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 3)')
                        self.state = 420
                        self.match(AutolevParser.T__2)
                        self.state = 421
                        self.expr(4)
                        pass
                    elif la_ == 5:
                        localctx = AutolevParser.ColonContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 422
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 2)')
                        self.state = 423
                        self.match(AutolevParser.T__15)
                        self.state = 424
                        self.expr(3)
                        pass
                self.state = 429
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 49, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    def sempred(self, localctx: RuleContext, ruleIndex: int, predIndex: int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[27] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception('No predicate with index:' + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx: ExprContext, predIndex: int):
        if predIndex == 0:
            return self.precpred(self._ctx, 16)
        if predIndex == 1:
            return self.precpred(self._ctx, 15)
        if predIndex == 2:
            return self.precpred(self._ctx, 14)
        if predIndex == 3:
            return self.precpred(self._ctx, 3)
        if predIndex == 4:
            return self.precpred(self._ctx, 2)