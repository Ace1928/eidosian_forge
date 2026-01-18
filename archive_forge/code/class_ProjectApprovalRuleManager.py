from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectApprovalRuleManager(ListMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/approval_rules'
    _obj_cls = ProjectApprovalRule
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('name', 'approvals_required'), optional=('user_ids', 'group_ids', 'protected_branch_ids', 'usernames'))