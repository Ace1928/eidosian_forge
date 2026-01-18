import json
from mlflow.environment_variables import MLFLOW_RUN_CONTEXT
from mlflow.tracking.context.abstract_context import RunContextProvider
class SystemEnvironmentContext(RunContextProvider):

    def in_context(self):
        return MLFLOW_RUN_CONTEXT.defined

    def tags(self):
        return json.loads(MLFLOW_RUN_CONTEXT.get())