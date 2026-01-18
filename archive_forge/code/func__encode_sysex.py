from .specs import CHANNEL_MESSAGES, MIN_PITCHWHEEL, SPEC_BY_TYPE
def _encode_sysex(msg):
    return [240] + list(msg['data']) + [247]