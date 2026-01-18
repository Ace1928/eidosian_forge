from textwrap import dedent
class DashException(Exception):

    def __init__(self, msg=''):
        super().__init__(dedent(msg).strip())