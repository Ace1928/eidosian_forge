import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowRunInputs(graphene.ObjectType):
    dataset_inputs = graphene.List(graphene.NonNull(MlflowDatasetInput))