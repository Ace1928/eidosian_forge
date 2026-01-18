from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
class SaySomethingFixed(ClientIDMutation):

    class Input:
        what = String()
    phrase = String()

    @staticmethod
    def mutate_and_get_payload(self, info, what, client_mutation_id=None):
        return FixedSaySomething(phrase=str(what))