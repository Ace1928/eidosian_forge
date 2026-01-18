from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
class SaySomething(ClientIDMutation):

    class Input:
        what = String()
    phrase = String()

    @staticmethod
    def mutate_and_get_payload(self, info, what, client_mutation_id=None):
        return SaySomething(phrase=str(what))