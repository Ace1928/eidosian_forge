from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.error.ErrorListener import ProxyErrorListener, ConsoleErrorListener
def addErrorListener(self, listener):
    self._listeners.append(listener)