import graphene
class CreatePostResult(graphene.Union):

    class Meta:
        types = [Success, Error]