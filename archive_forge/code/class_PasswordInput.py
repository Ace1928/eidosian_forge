import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class PasswordInput(TextInput):

    def _renderText(self):
        return '*' * len(self.buffer)