import copy
import functools
import queue
import warnings
import dogpile.cache
import keystoneauth1.exceptions
import keystoneauth1.session
import requests.models
import requestsexceptions
from openstack import _log
from openstack.cloud import _object_store
from openstack.cloud import _utils
from openstack.cloud import meta
import openstack.config
from openstack.config import cloud_region as cloud_region_mod
from openstack import exceptions
from openstack import proxy
from openstack import utils
def _get_project_info(self, project_id=None):
    project_info = utils.Munch(id=project_id, name=None, domain_id=None, domain_name=None)
    if not project_id or project_id == self.current_project_id:
        auth_args = self.config.config.get('auth', {})
        project_info['id'] = self.current_project_id
        project_info['name'] = auth_args.get('project_name')
        project_info['domain_id'] = auth_args.get('project_domain_id')
        project_info['domain_name'] = auth_args.get('project_domain_name')
    return project_info