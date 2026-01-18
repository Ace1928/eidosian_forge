from .specs import CHANNEL_MESSAGES, MIN_PITCHWHEEL, SPEC_BY_TYPE
def _encode_songpos(data):
    pos = data['pos']
    return [242, pos & 127, pos >> 7]