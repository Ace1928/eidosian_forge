from Xlib import X
from Xlib.protocol import rq, structs
class ChangeHosts(rq.Request):
    _request = rq.Struct(rq.Opcode(109), rq.Set('mode', 1, (X.HostInsert, X.HostDelete)), rq.RequestLength(), rq.Set('host_family', 1, (X.FamilyInternet, X.FamilyDECnet, X.FamilyChaos)), rq.Pad(1), rq.LengthOf('host', 2), rq.List('host', rq.Card8Obj))