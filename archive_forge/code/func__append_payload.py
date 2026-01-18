from enum import Enum
def _append_payload(self, payload_field):
    """Append PAYLOAD_FIELD."""
    if payload_field is not None:
        self.args.append('PAYLOAD_FIELD')
        self.args.append(payload_field)