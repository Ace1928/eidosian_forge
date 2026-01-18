from abc import ABCMeta, abstractmethod
from mlflow.utils.annotations import developer_stable

        Returns:
            True if this flavor has a `build_image` method defined for building a docker
            container capable of serving the model, False otherwise.
        