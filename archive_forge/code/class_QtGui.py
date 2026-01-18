import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QtGui(ProxyNamespace):

    class QIcon(ProxyClass):
        pass

    class QConicalGradient(ProxyClass):
        pass

    class QLinearGradient(ProxyClass):
        pass

    class QRadialGradient(ProxyClass):
        pass

    class QBrush(ProxyClass):
        pass

    class QPainter(ProxyClass):
        pass

    class QPalette(ProxyClass):
        pass

    class QFont(ProxyClass):
        pass