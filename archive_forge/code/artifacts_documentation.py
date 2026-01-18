from typing import Any, Callable, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
Download a single artifact file from a specific tag or branch from
        within the job's artifacts archive.

        Args:
            ref_name: Branch or tag name in repository. HEAD or SHA references
                are not supported.
            artifact_path: Path to a file inside the artifacts archive.
            job: The name of the job.
            streamed: If True the data will be processed by chunks of
                `chunk_size` and each chunk is passed to `action` for
                treatment
            iterator: If True directly return the underlying response
                iterator
            action: Callable responsible of dealing with chunk of
                data
            chunk_size: Size of each chunk
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the artifacts could not be retrieved

        Returns:
            The artifact if `streamed` is False, None otherwise.
        