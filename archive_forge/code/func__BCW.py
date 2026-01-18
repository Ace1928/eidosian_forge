from reportlab.lib.validators import isInt, isNumber, isString, isColorOrNone, isBoolean, EitherOr, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import rl_exec
from reportlab.graphics.shapes import Rect, Group, String
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode.widgets import BarcodeStandard93
def _BCW(doc, codeName, attrMap, mod, value, **kwds):
    """factory for Barcode Widgets"""
    _pre_init = kwds.pop('_pre_init', '')
    _methods = kwds.pop('_methods', '')
    name = 'Barcode' + codeName
    ns = vars().copy()
    code = 'from %s import %s' % (mod, codeName)
    rl_exec(code, ns)
    ns['_BarcodeWidget'] = _BarcodeWidget
    ns['doc'] = "\n\t'''%s'''" % doc if doc else ''
    code = 'class %(name)s(_BarcodeWidget,%(codeName)s):%(doc)s\n\t_BCC = %(codeName)s\n\tcodeName = %(codeName)r\n\tdef __init__(self,**kw):%(_pre_init)s\n\t\t_BarcodeWidget.__init__(self,%(value)r,**kw)%(_methods)s' % ns
    rl_exec(code, ns)
    Klass = ns[name]
    if attrMap:
        Klass._attrMap = attrMap
    for k, v in kwds.items():
        setattr(Klass, k, v)
    return Klass