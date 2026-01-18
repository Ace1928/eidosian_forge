from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
def _what(data):
    for rule in _rules:
        if (res := rule(data)):
            return res
    else:
        return None