from .specs import CHANNEL_MESSAGES, MIN_PITCHWHEEL, SPEC_BY_TYPE
def _encode_quarter_frame(msg):
    return [241, msg['frame_type'] << 4 | msg['frame_value']]