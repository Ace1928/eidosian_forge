from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
class BaseCreateUser(Mutation):

    class Arguments:
        name = String()
    name = String()

    def mutate(self, info, **args):
        return args