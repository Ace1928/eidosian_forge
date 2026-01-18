from abc import ABCMeta, abstractmethod
from mlflow.utils.annotations import developer_stable

        Submit an entrypoint. It must return a SubmittedRun object to track the execution

        Args:
            project_uri: URI of the project to execute, e.g. a local filesystem path
                or a Git repository URI like https://github.com/mlflow/mlflow-example
            entry_point: Entry point to run within the project.
            params: Dict of parameters to pass to the entry point
            version: For git-based projects, either a commit hash or a branch name.
            backend_config: A dictionary, or a path to a JSON file (must end in '.json'), which
                will be passed as config to the backend. The exact content which
                should be provided is different for each execution backend and is
                documented at https://www.mlflow.org/docs/latest/projects.html.
            tracking_uri: URI of tracking server against which to log run information related
                to project execution.
            experiment_id: ID of experiment under which to launch the run.

        Returns:
            A :py:class:`mlflow.projects.SubmittedRun`. This function is expected to run
            the project asynchronously, i.e. it should trigger project execution and then
            immediately return a `SubmittedRun` to track execution status.
        