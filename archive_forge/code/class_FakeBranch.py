import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
class FakeBranch:

    def get_config_stack(self):
        return self