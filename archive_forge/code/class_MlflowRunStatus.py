import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowRunStatus(graphene.Enum):
    RUNNING = 1
    SCHEDULED = 2
    FINISHED = 3
    FAILED = 4
    KILLED = 5