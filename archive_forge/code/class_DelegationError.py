from .trait_base import class_of
class DelegationError(TraitError):

    def __init__(self, args):
        self.args = (args,)