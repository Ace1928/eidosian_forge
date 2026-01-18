import graphene
class MyInput(MyInputClass):

    class Meta:
        fields = dict(x=graphene.Field(graphene.Int))