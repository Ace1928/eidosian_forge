from antlr4 import *
from io import StringIO
import sys
class AutolevLexer(Lexer):
    atn = ATNDeserializer().deserialize(serializedATN())
    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]
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
    channelNames = [u'DEFAULT_TOKEN_CHANNEL', u'HIDDEN']
    modeNames = ['DEFAULT_MODE']
    literalNames = ['<INVALID>', "'['", "']'", "'='", "'+='", "'-='", "':='", "'*='", "'/='", "'^='", "','", "'''", "'('", "')'", "'{'", "'}'", "':'", "'+'", "'-'", "';'", "'.'", "'>'", "'0>'", "'1>>'", "'^'", "'*'", "'/'"]
    symbolicNames = ['<INVALID>', 'Mass', 'Inertia', 'Input', 'Output', 'Save', 'UnitSystem', 'Encode', 'Newtonian', 'Frames', 'Bodies', 'Particles', 'Points', 'Constants', 'Specifieds', 'Imaginary', 'Variables', 'MotionVariables', 'INT', 'FLOAT', 'EXP', 'LINE_COMMENT', 'ID', 'WS']
    ruleNames = ['T__0', 'T__1', 'T__2', 'T__3', 'T__4', 'T__5', 'T__6', 'T__7', 'T__8', 'T__9', 'T__10', 'T__11', 'T__12', 'T__13', 'T__14', 'T__15', 'T__16', 'T__17', 'T__18', 'T__19', 'T__20', 'T__21', 'T__22', 'T__23', 'T__24', 'T__25', 'Mass', 'Inertia', 'Input', 'Output', 'Save', 'UnitSystem', 'Encode', 'Newtonian', 'Frames', 'Bodies', 'Particles', 'Points', 'Constants', 'Specifieds', 'Imaginary', 'Variables', 'MotionVariables', 'DIFF', 'DIGIT', 'INT', 'FLOAT', 'EXP', 'LINE_COMMENT', 'ID', 'WS']
    grammarFileName = 'Autolev.g4'

    def __init__(self, input=None, output: TextIO=sys.stdout):
        super().__init__(input, output)
        self.checkVersion('4.11.1')
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None