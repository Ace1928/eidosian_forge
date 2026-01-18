from .specs import CHANNEL_MESSAGES, MIN_PITCHWHEEL, SPEC_BY_TYPE
def _encode_note_off(msg):
    return [128 | msg['channel'], msg['note'], msg['velocity']]