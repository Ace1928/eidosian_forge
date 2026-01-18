import datetime
import graphene
from graphql.utilities import print_schema
class SetDatetime(graphene.Mutation):

    class Arguments:
        filters = Filters(required=True)
    ok = graphene.Boolean()

    def mutate(root, info, filters):
        return SetDatetime(ok=True)