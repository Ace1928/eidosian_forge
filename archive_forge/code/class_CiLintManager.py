from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.cli import register_custom_action
from gitlab.exceptions import GitlabCiLintError
from gitlab.mixins import CreateMixin, GetWithoutIdMixin
from gitlab.types import RequiredOptional
class CiLintManager(CreateMixin, RESTManager):
    _path = '/ci/lint'
    _obj_cls = CiLint
    _create_attrs = RequiredOptional(required=('content',), optional=('include_merged_yaml', 'include_jobs'))

    @register_custom_action('CiLintManager', ('content',), optional=('include_merged_yaml', 'include_jobs'))
    def validate(self, *args: Any, **kwargs: Any) -> None:
        """Raise an error if the CI Lint results are not valid.

        This is a custom python-gitlab method to wrap lint endpoints."""
        result = self.create(*args, **kwargs)
        if result.status != 'valid':
            message = ',\n'.join(result.errors)
            raise GitlabCiLintError(message)