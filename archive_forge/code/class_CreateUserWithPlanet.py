from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
class CreateUserWithPlanet(BaseCreateUser):

    class Arguments(BaseCreateUser.Arguments):
        planet = String()
    planet = String()

    def mutate(self, info, **args):
        return CreateUserWithPlanet(**args)