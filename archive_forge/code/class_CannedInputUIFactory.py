import warnings
class CannedInputUIFactory(SilentUIFactory):
    """A silent UI that return canned input."""

    def __init__(self, responses):
        self.responses = responses

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.responses)

    def confirm_action(self, prompt, confirmation_id, args):
        return self.get_boolean(prompt % args)

    def get_boolean(self, prompt):
        return self.responses.pop(0)

    def get_integer(self, prompt):
        return self.responses.pop(0)

    def get_password(self, prompt='', **kwargs):
        return self.responses.pop(0)

    def get_username(self, prompt, **kwargs):
        return self.responses.pop(0)

    def assert_all_input_consumed(self):
        if self.responses:
            raise AssertionError('expected all input in %r to be consumed' % (self,))