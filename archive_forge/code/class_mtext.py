import docutils.utils.math.tex2unichar as tex2unichar
class mtext(math):
    nchildren = 0

    def __init__(self, text):
        self.text = text

    def xml_body(self):
        return [self.text]