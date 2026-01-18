from .specs import CHANNEL_MESSAGES, MIN_PITCHWHEEL, SPEC_BY_TYPE
def _encode_note_on(msg):
    return [144 | msg['channel'], msg['note'], msg['velocity']]