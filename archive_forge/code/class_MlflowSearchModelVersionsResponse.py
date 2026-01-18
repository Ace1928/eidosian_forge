import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowSearchModelVersionsResponse(graphene.ObjectType):
    model_versions = graphene.List(graphene.NonNull(MlflowModelVersion))
    next_page_token = graphene.String()