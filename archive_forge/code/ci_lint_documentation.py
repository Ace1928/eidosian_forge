from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.cli import register_custom_action
from gitlab.exceptions import GitlabCiLintError
from gitlab.mixins import CreateMixin, GetWithoutIdMixin
from gitlab.types import RequiredOptional
Raise an error if the Project CI Lint results are not valid.

        This is a custom python-gitlab method to wrap lint endpoints.