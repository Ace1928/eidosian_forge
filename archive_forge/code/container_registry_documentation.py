from typing import Any, cast, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
Delete Tag in bulk

        Args:
            name_regex_delete: The regex of the name to delete. To delete all
                tags specify .*.
            keep_n: The amount of latest tags of given name to keep.
            name_regex_keep: The regex of the name to keep. This value
                overrides any matches from name_regex.
            older_than: Tags to delete that are older than the given time,
                written in human readable form 1h, 1d, 1month.
            **kwargs: Extra options to send to the server (e.g. sudo)
        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server cannot perform the request
        