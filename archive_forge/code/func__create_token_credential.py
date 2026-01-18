from castellan.common.credentials import token
from castellan.tests import base
def _create_token_credential(self):
    return token.Token(self.token)