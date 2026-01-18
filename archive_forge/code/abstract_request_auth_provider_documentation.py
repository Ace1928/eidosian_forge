from abc import ABC, abstractmethod
from mlflow.utils.annotations import developer_stable

        Generate request auth object (e.g., `requests.auth import HTTPBasicAuth`). See
        https://requests.readthedocs.io/en/latest/user/authentication/ for more details.

        Returns:
            request auth object.
        