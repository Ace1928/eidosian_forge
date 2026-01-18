from Xlib import X
from Xlib.protocol import rq, structs
class ConfigureWindow(rq.Request):
    _request = rq.Struct(rq.Opcode(12), rq.Pad(1), rq.RequestLength(), rq.Window('window'), rq.ValueList('attrs', 2, 2, rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Int16('border_width'), rq.Window('sibling'), rq.Set('stack_mode', 1, (X.Above, X.Below, X.TopIf, X.BottomIf, X.Opposite))))