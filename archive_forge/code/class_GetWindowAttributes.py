from Xlib import X
from Xlib.protocol import rq, structs
class GetWindowAttributes(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(3), rq.Pad(1), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('backing_store'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('visual'), rq.Card16('win_class'), rq.Card8('bit_gravity'), rq.Card8('win_gravity'), rq.Card32('backing_bit_planes'), rq.Card32('backing_pixel'), rq.Card8('save_under'), rq.Card8('map_is_installed'), rq.Card8('map_state'), rq.Card8('override_redirect'), rq.Colormap('colormap', (X.NONE,)), rq.Card32('all_event_masks'), rq.Card32('your_event_mask'), rq.Card16('do_not_propagate_mask'), rq.Pad(2))