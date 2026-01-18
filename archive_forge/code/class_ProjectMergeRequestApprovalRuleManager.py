from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectMergeRequestApprovalRuleManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/approval_rules'
    _obj_cls = ProjectMergeRequestApprovalRule
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}
    _update_attrs = RequiredOptional(required=('id', 'merge_request_iid', 'approval_rule_id', 'name', 'approvals_required'), optional=('user_ids', 'group_ids'))
    _create_attrs = RequiredOptional(required=('id', 'merge_request_iid', 'name', 'approvals_required'), optional=('approval_project_rule_id', 'user_ids', 'group_ids'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestApprovalRule:
        return cast(ProjectMergeRequestApprovalRule, super().get(id=id, lazy=lazy, **kwargs))

    def create(self, data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> RESTObject:
        """Create a new object.

        Args:
            data: Parameters to send to the server to create the
                         resource
            **kwargs: Extra options to send to the server (e.g. sudo or
                      'ref_name', 'stage', 'name', 'all')

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server cannot perform the request

        Returns:
            A new instance of the manage object class build with
                the data sent by the server
        """
        if TYPE_CHECKING:
            assert data is not None
        new_data = data.copy()
        new_data['id'] = self._from_parent_attrs['project_id']
        new_data['merge_request_iid'] = self._from_parent_attrs['mr_iid']
        return CreateMixin.create(self, new_data, **kwargs)