import warnings
class NoninteractiveUIFactory(UIFactory):
    """Base class for UIs with no user."""

    def confirm_action(self, prompt, confirmation_id, prompt_kwargs):
        return True

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)