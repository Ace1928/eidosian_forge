from abc import ABCMeta, abstractmethod
from mlflow.utils.annotations import developer_stable
Provide the MLflow Experiment ID for the current MLflow client context.

        Assumes that ``in_context()`` is ``True``.

        Returns:
            The ID of the MLflow Experiment associated with the current context.

        